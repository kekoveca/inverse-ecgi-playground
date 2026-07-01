#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np


# Allow running as:
#   python3 examples/full_inverse_experiment_torso.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import RelativeGaussianNoise, select_farthest_point_electrodes, select_random_electrodes
from fem import NeumannPoissonSolver
from forward import (
    ForwardSolver,
    export_dolfinx_function_to_vtx,
    export_electrode_markers_to_vtx,
    export_forward_result_to_vtx,
    inspect_electrode_marker_mapping,
)
from geometry import ElectrodeSet, SourceRegion, electrode_placement_report, read_gmsh_meshio
from green import (
    GreenSolver,
    build_green_transfer_matrix,
    check_measurement_matrix_compatibility,
    compare_forward_and_green,
    measurement_matrix_row_sums,
)
from inverse import SingleDipoleInverseSolver, inverse_reconstruction_metrics
from measurements import build_measurement_operator, central_project_electrodes_to_surface
from sources import (
    PointDipole,
    assemble_point_dipole_rhs_petsc,
    create_cell_marker_function,
    inspect_point_dipole_location_petsc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full forward -> Green -> inverse tutorial experiment on torso.msh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mesh", default="torso.msh", help="Path to Gmsh .msh torso mesh.")
    parser.add_argument("--output", default="output/full_inverse_experiment", help="Output directory.")
    parser.add_argument("--domain-name", default="domain", help="Physical group name for tetra volume.")
    parser.add_argument("--boundary-name", default="boundary", help="Physical group name for triangle boundary.")
    parser.add_argument("--num-electrodes", type=int, default=32, help="Number of demo electrodes.")
    parser.add_argument("--num-candidates", type=int, default=50, help="Number of source candidates.")
    parser.add_argument(
        "--electrode-offset",
        type=float,
        default=0.0,
        help="Reserved for projection APIs with normal offsets; current central projection ignores it.",
    )
    parser.add_argument(
        "--electrode-mode",
        choices=["surface-farthest", "surface-random", "existing-api-default"],
        default="surface-farthest",
        help="How to select initial demo electrode points on the surface.",
    )
    parser.add_argument("--moment", nargs=3, type=float, default=[0.0, 0.0, 1.0], help="True dipole moment.")
    parser.add_argument("--source-index", type=int, default=None, help="True source candidate index.")
    parser.add_argument("--snr-db", type=float, default=None, help="Optional relative Gaussian noise SNR in dB.")
    parser.add_argument("--lambda-reg", type=float, default=1e-10, help="Tikhonov regularization for inverse LS.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Constant scalar conductivity.")
    parser.add_argument(
        "--reference",
        choices=["average", "single", "none"],
        default="average",
        help="Electrode reference system. Green solves usually require average or single.",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=None,
        help="Electrode index required when --reference single is used.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible demo choices.")
    parser.add_argument("--no-export", action="store_true", help="Skip VTX/BP ParaView exports.")
    parser.add_argument(
        "--green-sign-check",
        dest="green_sign_check",
        action="store_true",
        default=True,
        help="Run forward/Green sign consistency diagnostics.",
    )
    parser.add_argument(
        "--skip-green-sign-check",
        dest="green_sign_check",
        action="store_false",
        help="Skip forward/Green sign consistency diagnostics.",
    )
    return parser.parse_args()


def require_dolfinx_runtime() -> None:
    """Fail early with a clear message if the DOLFINx stack is unavailable."""
    try:
        import dolfinx  # noqa: F401
        import mpi4py  # noqa: F401
        import petsc4py  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "This example requires DOLFINx, mpi4py and petsc4py. "
            "Run it inside the environment where the FEM tests pass."
        ) from exc


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def surface_used_vertex_ids(surface_mesh) -> np.ndarray:
    """Return unique point ids actually referenced by surface triangles."""
    cells = np.asarray(surface_mesh.cells, dtype=np.int64)
    if cells.size == 0:
        return np.array([], dtype=np.int64)
    return np.unique(cells.reshape(-1)).astype(np.int64)


def mesh_diagnostics_summary(volume_mesh, surface_mesh) -> dict[str, int]:
    """Return compact mesh-size diagnostics for the tutorial report."""
    return {
        "num_volume_points": int(volume_mesh.num_points),
        "num_volume_cells": int(volume_mesh.num_cells),
        "num_surface_point_array_size": int(surface_mesh.num_points),
        "num_surface_cells": int(surface_mesh.num_cells),
        "num_surface_used_vertices": int(surface_used_vertex_ids(surface_mesh).size),
    }


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 0.0:
        return float(np.linalg.norm(point - a))
    t = np.clip(float(np.dot(point - a, ab)) / denom, 0.0, 1.0)
    projection = a + t * ab
    return float(np.linalg.norm(point - projection))


def _point_triangle_distance(point: np.ndarray, triangle: np.ndarray) -> float:
    """Distance from point to triangle using a standard closest-point test."""
    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        denom = d1 - d3
        if abs(denom) > 0.0:
            v = d1 / denom
            return float(np.linalg.norm(point - (a + v * ab)))

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        denom = d2 - d6
        if abs(denom) > 0.0:
            w = d2 / denom
            return float(np.linalg.norm(point - (a + w * ac)))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom = (d4 - d3) + (d5 - d6)
        if abs(denom) > 0.0:
            w = (d4 - d3) / denom
            return float(np.linalg.norm(point - (b + w * (c - b))))

    normal = np.cross(ab, ac)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 0.0:
        return min(
            _point_segment_distance(point, a, b),
            _point_segment_distance(point, b, c),
            _point_segment_distance(point, c, a),
        )
    return abs(float(np.dot(point - a, normal))) / normal_norm


def inspect_electrodes_nearest_surface_cells(surface_mesh, electrodes, centroid_candidates: int = 32) -> dict[str, Any]:
    """Find nearest surface triangle diagnostics for every electrode.

    The search uses triangle centroids to choose a small candidate set and then
    computes exact point-to-triangle distances for those candidates. This is a
    reporting helper for the example, not a production projection routine.
    """
    if surface_mesh.cell_type != "triangle":
        raise ValueError("surface_mesh must contain triangle cells")
    triangles = np.asarray(surface_mesh.points[surface_mesh.cells], dtype=float)
    if triangles.size == 0:
        raise ValueError("surface mesh must contain at least one triangle")
    positions = np.asarray(electrodes.positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("electrodes.positions must have shape (n_electrodes, 3)")

    centroids = triangles.mean(axis=1)
    k = max(1, min(int(centroid_candidates), triangles.shape[0]))
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(centroids)
        _, candidate_ids = tree.query(positions, k=k)
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        if candidate_ids.ndim == 1:
            candidate_ids = candidate_ids[:, None]
    except ImportError:  # pragma: no cover - scipy is usually available in this project
        centroid_distances = np.linalg.norm(centroids[None, :, :] - positions[:, None, :], axis=2)
        candidate_ids = np.argsort(centroid_distances, axis=1)[:, :k].astype(np.int64)

    nearest_cell_ids = np.full(positions.shape[0], -1, dtype=np.int64)
    nearest_distances = np.full(positions.shape[0], np.inf, dtype=float)
    for electrode_id, position in enumerate(positions):
        for cell_id in np.atleast_1d(candidate_ids[electrode_id]):
            distance = _point_triangle_distance(position, triangles[int(cell_id)])
            if distance < nearest_distances[electrode_id]:
                nearest_distances[electrode_id] = distance
                nearest_cell_ids[electrode_id] = int(cell_id)

    missing = nearest_cell_ids < 0
    finite_distances = nearest_distances[np.isfinite(nearest_distances)]
    return {
        "num_electrodes": int(positions.shape[0]),
        "nearest_surface_cell_ids": nearest_cell_ids,
        "nearest_surface_distances": nearest_distances,
        "max_nearest_surface_distance": float(finite_distances.max()) if finite_distances.size else float("nan"),
        "mean_nearest_surface_distance": float(finite_distances.mean()) if finite_distances.size else float("nan"),
        "num_nearest_surface_cell_ids_missing": int(np.count_nonzero(missing)),
    }


def read_torso_mesh(mesh_path: Path, domain_name: str, boundary_name: str):
    """Read torso.msh and extract volume/surface physical groups."""
    if not mesh_path.exists():
        raise FileNotFoundError(f"mesh file does not exist: {mesh_path}")

    tagged = read_gmsh_meshio(mesh_path, dim=3)
    print("Physical groups / field_data:")
    for name, pair in sorted(tagged.field_data.items()):
        print(f"  {name}: dim={pair[0]}, tag={pair[1]}")

    try:
        volume_mesh = tagged.to_mesh_data(cell_type="tetra", physical_name=domain_name)
        surface_mesh = tagged.to_mesh_data(cell_type="triangle", physical_name=boundary_name)
    except KeyError as exc:
        available = ", ".join(sorted(tagged.field_data))
        raise KeyError(
            f"Could not extract requested physical group. Available physical groups: {available}"
        ) from exc

    if volume_mesh.cell_type != "tetra":
        raise ValueError("volume_mesh must contain tetra cells")
    if surface_mesh.cell_type != "triangle":
        raise ValueError("surface_mesh must contain triangle cells")

    bbox_min, bbox_max = volume_mesh.bounding_box()
    mesh_summary = mesh_diagnostics_summary(volume_mesh, surface_mesh)
    print("Volume mesh:")
    print(f"  point array size: {mesh_summary['num_volume_points']}")
    print(f"  tetra cells: {mesh_summary['num_volume_cells']}")
    print("Surface mesh:")
    print(f"  point array size: {mesh_summary['num_surface_point_array_size']}")
    print(f"  triangle cells: {mesh_summary['num_surface_cells']}")
    print(f"  used surface vertices: {mesh_summary['num_surface_used_vertices']}")
    print(f"Volume bbox min: {bbox_min}")
    print(f"Volume bbox max: {bbox_max}")
    return tagged, volume_mesh, surface_mesh


def unique_surface_vertex_electrodes(surface_mesh, *, name: str = "surface_vertex_candidates") -> ElectrodeSet:
    """Build demo electrode candidates from unique vertices used by the surface mesh."""
    surface_node_ids = np.unique(surface_mesh.cells.reshape(-1)).astype(np.int64)
    positions = surface_mesh.points[surface_node_ids]
    labels = [f"S{i:04d}" for i in range(positions.shape[0])]
    return ElectrodeSet(
        positions=positions,
        labels=labels,
        name=name,
        metadata={"source": "surface_mesh_vertices", "surface_node_ids": surface_node_ids.tolist()},
    )


def build_demo_surface_electrodes(
    volume_mesh,
    surface_mesh,
    num_electrodes: int,
    mode: str,
    seed: int,
    offset: float,
) -> tuple[ElectrodeSet, dict[str, Any], dict[str, Any]]:
    """Select demo surface electrodes and run the existing projection API.

    The selection is intentionally simple and tutorial-oriented. The projection
    itself uses ``central_project_electrodes_to_surface`` from ``measurements``.
    """
    num_electrodes = int(num_electrodes)
    if num_electrodes < 1:
        raise ValueError("--num-electrodes must be positive")
    if offset != 0.0:
        warnings.warn(
            "--electrode-offset was provided, but the current production central projection API "
            "does not support normal offsets; the offset is not applied.",
            RuntimeWarning,
        )

    candidates = unique_surface_vertex_electrodes(surface_mesh)
    if num_electrodes > candidates.num_electrodes:
        raise ValueError(
            f"requested {num_electrodes} electrodes, but only {candidates.num_electrodes} "
            "unique surface vertices are available"
        )

    if mode in {"surface-farthest", "existing-api-default"}:
        subset = select_farthest_point_electrodes(
            candidates,
            n=num_electrodes,
            seed=seed,
            name=f"surface_farthest_{num_electrodes}",
        )
    elif mode == "surface-random":
        subset = select_random_electrodes(
            candidates,
            n=num_electrodes,
            seed=seed,
            name=f"surface_random_{num_electrodes}",
        )
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"unsupported electrode mode {mode!r}")

    projected_electrodes, projection_report = central_project_electrodes_to_surface(
        volume_mesh=volume_mesh,
        electrodes=subset.electrodes,
        surface_mesh=surface_mesh,
    )
    projection_summary = projection_report.to_summary_dict()
    projection_summary.update(
        {
            "selection_mode": mode,
            "selection_metadata": subset.metadata,
            "production_projection_api": "measurements.central_project_electrodes_to_surface",
            "offset_requested": float(offset),
            "offset_applied": False,
            "num_unchanged": int(projected_electrodes.num_electrodes - projection_report.num_projected),
            "num_projection_surface_cell_ids_missing": int(np.count_nonzero(projection_report.surface_cell_ids < 0)),
        }
    )
    surface_diagnostics = inspect_electrodes_nearest_surface_cells(surface_mesh, projected_electrodes)
    print(f"Electrodes: {projected_electrodes.num_electrodes}")
    print("Projection:")
    print(f"  production API: {projection_summary['production_projection_api']}")
    print(f"  only_outside_electrodes: {projection_summary['metadata'].get('only_outside_electrodes')}")
    print(f"  num projected: {projection_summary['num_projected']}")
    print(f"  num unchanged: {projection_summary['num_unchanged']}")
    print(f"  offset requested: {projection_summary['offset_requested']}")
    print(f"  offset applied: {projection_summary['offset_applied']}")
    print("First electrodes:")
    for label, position in zip(projected_electrodes.labels[:5], projected_electrodes.positions[:5]):
        print(f"  {label}: {position}")

    placement = electrode_placement_report(projected_electrodes, surface_mesh)
    electrodes_on_surface_nodes = bool(placement.max_distance_to_nearest_node <= 1e-10)
    projection_summary["electrodes_exactly_on_surface_nodes"] = electrodes_on_surface_nodes
    print(
        "Electrode nearest-surface-node distance: "
        f"mean={placement.mean_distance_to_nearest_node:.6g}, "
        f"max={placement.max_distance_to_nearest_node:.6g}"
    )
    print("Surface location:")
    print(
        "  nearest surface cell ids available: "
        f"{'yes' if surface_diagnostics['num_nearest_surface_cell_ids_missing'] == 0 else 'no'}"
    )
    print(f"  nearest surface distance mean: {surface_diagnostics['mean_nearest_surface_distance']:.6g}")
    print(f"  nearest surface distance max: {surface_diagnostics['max_nearest_surface_distance']:.6g}")
    print(f"  electrodes exactly on surface nodes: {'yes' if electrodes_on_surface_nodes else 'no'}")
    return projected_electrodes, projection_summary, surface_diagnostics


def build_source_candidates(volume_mesh, num_candidates: int, seed: int):
    """Build candidate source points from cell centers inside a central bbox."""
    num_candidates = int(num_candidates)
    if num_candidates < 1:
        raise ValueError("--num-candidates must be positive")

    bbox_min, bbox_max = volume_mesh.bounding_box()
    center = 0.5 * (bbox_min + bbox_max)
    half_size = 0.15 * (bbox_max - bbox_min)
    region = SourceRegion.from_bounding_box(
        volume_mesh,
        center - half_size,
        center + half_size,
        name="central_bbox_source_region",
        mode="center",
    )
    if region.num_candidates == 0:
        warnings.warn(
            "Central bounding-box source region is empty; falling back to all cell centers.",
            RuntimeWarning,
        )
        region = SourceRegion.all_cells(volume_mesh, name="all_cells_source_region")
    if region.num_candidates == 0:
        raise ValueError("source region is empty; cannot run inverse experiment")

    if region.num_candidates <= num_candidates:
        selected = np.arange(region.num_candidates, dtype=np.int64)
        if region.num_candidates < num_candidates:
            warnings.warn(
                f"Requested {num_candidates} candidates but only {region.num_candidates} are available.",
                RuntimeWarning,
            )
    else:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(region.num_candidates, size=num_candidates, replace=False)).astype(np.int64)

    candidate_points = region.candidate_points[selected]
    meshdata_cell_ids = region.candidate_cell_ids[selected]
    print(f"Source candidates: {candidate_points.shape[0]} from region {region.name!r}")
    print("Note: SourceRegion.candidate_cell_ids are MeshData cell ids. Green will locate DOLFINx cells from points.")
    return candidate_points, meshdata_cell_ids, region


def choose_true_source_index(candidate_points: np.ndarray, source_index: int | None, volume_mesh) -> int:
    if candidate_points.shape[0] == 0:
        raise ValueError("candidate_points must contain at least one point")
    if source_index is not None:
        source_index = int(source_index)
        if source_index < 0 or source_index >= candidate_points.shape[0]:
            raise IndexError("--source-index is outside selected candidate_points")
        return source_index
    bbox_min, bbox_max = volume_mesh.bounding_box()
    bbox_center = 0.5 * (bbox_min + bbox_max)
    return int(np.argmin(np.linalg.norm(candidate_points - bbox_center, axis=1)))


def save_electrodes_csv(path: Path, electrodes: ElectrodeSet) -> None:
    rows = [
        {"label": label, "x": pos[0], "y": pos[1], "z": pos[2]}
        for label, pos in zip(electrodes.labels, electrodes.positions)
    ]
    write_csv(path, rows, ["label", "x", "y", "z"])


def save_candidates_csv(path: Path, candidate_points: np.ndarray, dolfinx_cell_ids: np.ndarray | None = None) -> None:
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(candidate_points):
        row = {
            "candidate_index": index,
            "x": point[0],
            "y": point[1],
            "z": point[2],
        }
        if dolfinx_cell_ids is not None:
            row["dolfinx_cell_id"] = int(dolfinx_cell_ids[index])
        rows.append(row)
    fields = ["candidate_index", "x", "y", "z"]
    if dolfinx_cell_ids is not None:
        fields.append("dolfinx_cell_id")
    write_csv(path, rows, fields)


def save_electrode_marker_mapping_csv(path: Path, electrodes: ElectrodeSet, marker_info: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    nearest_dof_ids = marker_info["nearest_dof_ids"]
    nearest_dof_coordinates = marker_info["nearest_dof_coordinates"]
    nearest_distances = marker_info["nearest_distances"]
    for electrode_index, (label, position) in enumerate(zip(electrodes.labels, electrodes.positions)):
        nearest = nearest_dof_coordinates[electrode_index]
        rows.append(
            {
                "electrode_index": electrode_index,
                "label": label,
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "nearest_dof_id": int(nearest_dof_ids[electrode_index]),
                "nearest_dof_x": nearest[0],
                "nearest_dof_y": nearest[1],
                "nearest_dof_z": nearest[2],
                "distance": nearest_distances[electrode_index],
            }
        )
    write_csv(
        path,
        rows,
        [
            "electrode_index",
            "label",
            "x",
            "y",
            "z",
            "nearest_dof_id",
            "nearest_dof_x",
            "nearest_dof_y",
            "nearest_dof_z",
            "distance",
        ],
    )


def save_electrode_surface_diagnostics_csv(
    path: Path,
    electrodes: ElectrodeSet,
    projection_summary: dict[str, Any],
    surface_diagnostics: dict[str, Any],
) -> None:
    projected_indices = set(int(index) for index in projection_summary.get("projected_indices", []))
    projection_surface_cell_ids = np.asarray(
        projection_summary.get("surface_cell_ids", np.full(electrodes.num_electrodes, -1)),
        dtype=np.int64,
    )
    if projection_surface_cell_ids.shape != (electrodes.num_electrodes,):
        projection_surface_cell_ids = np.full(electrodes.num_electrodes, -1, dtype=np.int64)

    nearest_surface_cell_ids = np.asarray(surface_diagnostics["nearest_surface_cell_ids"], dtype=np.int64)
    nearest_surface_distances = np.asarray(surface_diagnostics["nearest_surface_distances"], dtype=float)
    rows: list[dict[str, Any]] = []
    for electrode_index, (label, position) in enumerate(zip(electrodes.labels, electrodes.positions)):
        rows.append(
            {
                "electrode_index": electrode_index,
                "label": label,
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "was_projected": electrode_index in projected_indices,
                "projection_surface_cell_id": int(projection_surface_cell_ids[electrode_index]),
                "nearest_surface_cell_id": int(nearest_surface_cell_ids[electrode_index]),
                "nearest_surface_distance": nearest_surface_distances[electrode_index],
            }
        )
    write_csv(
        path,
        rows,
        [
            "electrode_index",
            "label",
            "x",
            "y",
            "z",
            "was_projected",
            "projection_surface_cell_id",
            "nearest_surface_cell_id",
            "nearest_surface_distance",
        ],
    )


def safe_vtx_export(description: str, callback, export_errors: list[str]) -> None:
    try:
        path = callback()
        print(f"Exported {description}: {path}")
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        message = f"{description} export skipped/failed: {exc}"
        export_errors.append(message)
        print(f"WARNING: {message}")


def run_full_inverse_experiment(args: argparse.Namespace, electrode_builder=build_demo_surface_electrodes) -> int:
    """Run the full tutorial pipeline with a configurable electrode builder."""
    require_dolfinx_runtime()
    mesh_path = Path(args.mesh)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Read torso mesh
    _, volume_mesh, surface_mesh = read_torso_mesh(mesh_path, args.domain_name, args.boundary_name)
    mesh_summary = mesh_diagnostics_summary(volume_mesh, surface_mesh)

    # 3. Build/project electrodes on torso surface
    electrodes, electrode_projection_summary, electrode_surface_diagnostics = electrode_builder(
        volume_mesh=volume_mesh,
        surface_mesh=surface_mesh,
        num_electrodes=args.num_electrodes,
        mode=args.electrode_mode,
        seed=args.seed,
        offset=args.electrode_offset,
    )

    # 4. Build source candidates inside torso
    candidate_points, meshdata_candidate_cell_ids, source_region = build_source_candidates(
        volume_mesh,
        num_candidates=args.num_candidates,
        seed=args.seed,
    )
    true_candidate_index = choose_true_source_index(candidate_points, args.source_index, volume_mesh)
    moment = np.asarray(args.moment, dtype=float)
    if moment.shape != (3,) or not np.all(np.isfinite(moment)) or np.linalg.norm(moment) <= 0.0:
        raise ValueError("--moment must be a finite nonzero 3-vector")
    source = PointDipole(
        position=candidate_points[true_candidate_index],
        moment=moment,
    )
    print(f"True candidate index: {true_candidate_index}")
    print(f"True source position: {source.position}")
    print(f"True source moment: {source.moment}")

    # 5. Create FEM solver
    # Pure Neumann problems have a constant nullspace; the solver removes the
    # incompatible RHS component and fixes the output gauge after each solve.
    solver = NeumannPoissonSolver(
        mesh=volume_mesh,
        degree=1,
        sigma=args.sigma,
    )

    export_errors: list[str] = []
    green_diagnostics: dict[str, Any] | None = None
    try:
        # 6. Forward solve
        # The measurement operator uses MeshData node ordering, while the FEM
        # potential is in DOLFINx dof ordering. ForwardSolver handles that map.
        measurement_operator = build_measurement_operator(
            mesh=volume_mesh,
            electrodes=electrodes,
            reference=args.reference,
            reference_index=args.reference_index,
            surface_mesh=surface_mesh,
        )
        forward = ForwardSolver(
            poisson_solver=solver,
            measurement_operator=measurement_operator,
            reference=args.reference,
        )
        forward_result = forward.solve(source)
        print(f"Raw measurement norm: {forward_result.raw_measurement_norm:.6g}")
        print(f"Referenced measurement norm: {forward_result.measurement_norm:.6g}")
        if args.reference == "average":
            print(f"Sum of average-referenced measurements: {forward_result.measurements.sum():.6g}")

        if args.snr_db is None:
            noisy_measurements = forward_result.measurements.copy()
            noise = np.zeros_like(noisy_measurements)
        else:
            noise_model = RelativeGaussianNoise(snr_db=args.snr_db, seed=args.seed)
            noisy_measurements, noise = noise_model.apply(forward_result.measurements)
            print(f"Added relative Gaussian noise: SNR={args.snr_db:g} dB, ||noise||={np.linalg.norm(noise):.6g}")

        # 7. Build Green basis
        # This solves one Neumann problem per measurement channel.
        row_sums = measurement_matrix_row_sums(measurement_operator)
        max_row_sum = float(np.max(np.abs(row_sums))) if row_sums.size else 0.0
        print(f"Max abs Green RHS row sum: {max_row_sum:.6g}")
        if not check_measurement_matrix_compatibility(measurement_operator):
            raise ValueError(
                "Measurement matrix rows are not compatible with pure Neumann Green solves. "
                "Use average reference or check electrode/reference setup."
            )

        print("Solving Green basis: one Neumann solve per measurement channel...")
        green_solver = GreenSolver(
            poisson_solver=solver,
            measurement_operator=measurement_operator,
            keep_functions=True,
        )
        green_basis = green_solver.solve_all()

        # 8. Build transfer matrix
        # Pass only candidate_points. SourceRegion.candidate_cell_ids are
        # MeshData cell ids, while GreenTransferMatrix stores DOLFINx cell ids.
        transfer = build_green_transfer_matrix(
            poisson_solver=solver,
            green_basis=green_basis,
            candidate_points=candidate_points,
        )
        print(f"Transfer tensor shape: {transfer.A.shape}")
        print(f"Transfer sign: {transfer.sign}")
        print(f"Candidate DOLFINx cell ids available: {transfer.candidate_cell_ids.shape}")

        if transfer.num_measurements != noisy_measurements.shape[0]:
            raise ValueError(
                f"measurement length {noisy_measurements.shape[0]} does not match "
                f"transfer.num_measurements={transfer.num_measurements}"
            )

        if args.green_sign_check:
            green_diagnostics = compare_forward_and_green(
                forward_result,
                transfer,
                candidate_index=true_candidate_index,
                moment=source.moment,
            )
            print("Forward/Green consistency:")
            print(f"  rel_error_plus: {green_diagnostics['rel_error_plus']:.6g}")
            print(f"  rel_error_minus: {green_diagnostics['rel_error_minus']:.6g}")
            print(f"  best_sign: {green_diagnostics['best_sign']}")
            print(f"  best_rel_error: {green_diagnostics['best_rel_error']:.6g}")
            if green_diagnostics["best_rel_error"] > 1e-4:
                print("WARNING: forward/Green consistency error is larger than expected; check sign/orderings.")

        # 9. Solve inverse problem
        inverse_solver = SingleDipoleInverseSolver(
            transfer_matrix=transfer,
            lambda_reg=args.lambda_reg,
            reference=args.reference,
        )
        inverse_result = inverse_solver.solve(noisy_measurements)

        # 10. Metrics and report
        metrics = inverse_reconstruction_metrics(
            inverse_result,
            true_position=source.position,
            true_moment=source.moment,
        )
        print("Inverse result:")
        print(f"  estimated candidate index: {inverse_result.best_candidate_index}")
        print(f"  estimated position: {inverse_result.estimated_position}")
        print(f"  localization error: {metrics['localization_error']:.6g}")
        print(f"  estimated moment: {inverse_result.estimated_moment}")
        print(f"  moment angle error [deg]: {metrics['moment_angle_error_deg']:.6g}")
        print(f"  moment relative error: {metrics['moment_relative_error']:.6g}")
        print(f"  residual norm: {inverse_result.residual_norm:.6g}")
        print(f"  relative residual: {inverse_result.relative_residual:.6g}")

        electrode_marker_info = inspect_electrode_marker_mapping(solver, electrodes)
        print("Electrode marker mapping:")
        print(f"  num electrodes: {electrode_marker_info['num_electrodes']}")
        print(f"  unique dofs: {electrode_marker_info['num_unique_dofs']}")
        print(f"  collisions: {electrode_marker_info['num_collisions']}")
        print(f"  max nearest-dof distance: {electrode_marker_info['max_distance']:.6g}")
        print(f"  mean nearest-dof distance: {electrode_marker_info['mean_distance']:.6g}")

        experiment_summary = {
            "mesh_path": str(mesh_path),
            "domain_name": args.domain_name,
            "boundary_name": args.boundary_name,
            **mesh_summary,
            "num_electrodes": electrodes.num_electrodes,
            "electrode_mode": args.electrode_mode,
            "electrode_offset": args.electrode_offset,
            "electrode_projection": electrode_projection_summary,
            "measurement_operator_projection": measurement_operator.metadata.get("electrode_projection"),
            "electrode_num_projected": electrode_projection_summary["num_projected"],
            "electrode_num_unchanged": electrode_projection_summary["num_unchanged"],
            "electrode_nearest_surface_distance_mean": electrode_surface_diagnostics[
                "mean_nearest_surface_distance"
            ],
            "electrode_nearest_surface_distance_max": electrode_surface_diagnostics["max_nearest_surface_distance"],
            "electrode_num_projection_surface_cell_ids_missing": electrode_projection_summary[
                "num_projection_surface_cell_ids_missing"
            ],
            "electrode_num_nearest_surface_cell_ids_missing": electrode_surface_diagnostics[
                "num_nearest_surface_cell_ids_missing"
            ],
            "electrode_marker_num_unique_dofs": electrode_marker_info["num_unique_dofs"],
            "electrode_marker_num_collisions": electrode_marker_info["num_collisions"],
            "electrode_marker_max_distance": electrode_marker_info["max_distance"],
            "electrode_marker_mean_distance": electrode_marker_info["mean_distance"],
            "num_candidates": int(candidate_points.shape[0]),
            "source_region_name": source_region.name,
            "reference": args.reference,
            "reference_index": args.reference_index,
            "sigma": args.sigma,
            "snr_db": args.snr_db,
            "lambda_reg": args.lambda_reg,
            "true_candidate_index": true_candidate_index,
            "estimated_candidate_index": inverse_result.best_candidate_index,
            "true_position": source.position,
            "estimated_position": inverse_result.estimated_position,
            "true_moment": source.moment,
            "estimated_moment": inverse_result.estimated_moment,
            "localization_error": metrics["localization_error"],
            "moment_angle_error_deg": metrics["moment_angle_error_deg"],
            "moment_relative_error": metrics["moment_relative_error"],
            "relative_residual": inverse_result.relative_residual,
            "green_forward_best_rel_error": None
            if green_diagnostics is None
            else green_diagnostics["best_rel_error"],
            "green_forward_best_sign": None if green_diagnostics is None else green_diagnostics["best_sign"],
            "export_errors": export_errors,
        }

        write_json(output_dir / "experiment_summary.json", experiment_summary)
        write_json(output_dir / "inverse_summary.json", inverse_result.to_summary_dict())
        np.savez_compressed(
            output_dir / "measurements.npz",
            clean_measurements=forward_result.measurements,
            noisy_measurements=noisy_measurements,
            noise=noise,
            electrode_positions=electrodes.positions,
            candidate_points=candidate_points,
            meshdata_candidate_cell_ids=meshdata_candidate_cell_ids,
            candidate_dolfinx_cell_ids=transfer.candidate_cell_ids,
            true_position=source.position,
            estimated_position=inverse_result.estimated_position,
            true_moment=source.moment,
            estimated_moment=inverse_result.estimated_moment,
        )
        save_electrodes_csv(output_dir / "electrodes.csv", electrodes)
        save_electrode_surface_diagnostics_csv(
            output_dir / "electrode_surface_diagnostics.csv",
            electrodes,
            electrode_projection_summary,
            electrode_surface_diagnostics,
        )
        save_electrode_marker_mapping_csv(output_dir / "electrode_marker_mapping.csv", electrodes, electrode_marker_info)
        save_candidates_csv(output_dir / "candidates.csv", candidate_points, transfer.candidate_cell_ids)
        print(f"Saved report files to: {output_dir}")

        # 11. ParaView export
        if not args.no_export:
            # electrodes.bp is a diagnostic nodal marker field. It marks the
            # nearest FEM dof to each electrode position, not an independent
            # point cloud. Use electrode_marker_mapping.csv for exact distances.
            safe_vtx_export(
                "electrodes",
                lambda: export_electrode_markers_to_vtx(
                    solver,
                    electrodes,
                    output_dir / "electrodes.bp",
                    value_mode="index",
                    name="electrodes",
                ),
                export_errors,
            )
            safe_vtx_export(
                "potential",
                lambda: export_forward_result_to_vtx(forward_result, output_dir / "potential.bp"),
                export_errors,
            )
            rhs = assemble_point_dipole_rhs_petsc(solver, source)
            safe_vtx_export(
                "rhs",
                lambda: export_dolfinx_function_to_vtx(rhs, output_dir / "rhs.bp", name="rhs"),
                export_errors,
            )
            source_info = inspect_point_dipole_location_petsc(solver, source)
            true_marker = create_cell_marker_function(
                solver,
                int(source_info["used_cell_id"]),
                name="true_source_marker",
            )
            safe_vtx_export(
                "true_source_marker",
                lambda: export_dolfinx_function_to_vtx(
                    true_marker,
                    output_dir / "true_source_marker.bp",
                    name="true_source_marker",
                ),
                export_errors,
            )
            if inverse_result.estimated_cell_id is None:
                print("WARNING: estimated_source_marker skipped because estimated_cell_id is None")
            else:
                estimated_marker = create_cell_marker_function(
                    solver,
                    int(inverse_result.estimated_cell_id),
                    name="estimated_source_marker",
                )
                safe_vtx_export(
                    "estimated_source_marker",
                    lambda: export_dolfinx_function_to_vtx(
                        estimated_marker,
                        output_dir / "estimated_source_marker.bp",
                        name="estimated_source_marker",
                    ),
                    export_errors,
                )
            # Update summary with export errors collected during VTX writes.
            experiment_summary["export_errors"] = export_errors
            write_json(output_dir / "experiment_summary.json", experiment_summary)
        else:
            print("ParaView export skipped by --no-export.")

    finally:
        # 12. Cleanup
        solver.destroy()

    return 0


def main() -> int:
    # 1. Parse arguments
    return run_full_inverse_experiment(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
