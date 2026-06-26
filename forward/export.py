from __future__ import annotations

from pathlib import Path

import numpy as np

from .result import ForwardResult


def _require_dolfinx_io():
    try:
        from dolfinx import io
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("exporting FEM potentials requires dolfinx") from exc
    return io


def _function_mesh(potential):
    function_space = getattr(potential, "function_space", None)
    mesh = getattr(function_space, "mesh", None)
    vector = getattr(potential, "x", None)
    if mesh is None:
        raise TypeError("potential must be a dolfinx.fem.Function with function_space.mesh")
    if vector is None or not hasattr(vector, "array"):
        raise TypeError("potential must be a dolfinx.fem.Function with x.array values")
    return mesh


def _solver_function_space(solver):
    V = getattr(solver, "V", None)
    if V is None:
        raise TypeError("solver must expose a DOLFINx scalar P1 FunctionSpace as solver.V")
    if not hasattr(V, "tabulate_dof_coordinates"):
        raise TypeError("solver.V must provide tabulate_dof_coordinates()")
    return V


def _nearest_dofs_for_points(dof_coordinates: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover - scipy is optional
        diff = dof_coordinates[None, :, :] - points[:, None, :]
        distances = np.linalg.norm(diff, axis=2)
        nearest_dof_ids = np.argmin(distances, axis=1).astype(np.int64)
        nearest_distances = distances[np.arange(points.shape[0]), nearest_dof_ids]
        return nearest_dof_ids, nearest_distances

    tree = cKDTree(dof_coordinates)
    nearest_distances, nearest_dof_ids = tree.query(points)
    return np.asarray(nearest_dof_ids, dtype=np.int64), np.asarray(nearest_distances, dtype=float)


def inspect_electrode_marker_mapping(solver, electrodes, tol: float | None = None) -> dict:
    """Map electrode positions to nearest FEM dofs for ParaView diagnostics.

    The returned mapping is diagnostic only: electrode positions are not moved,
    and the nearest dof marker is not an independent point-cloud export.
    """
    V = _solver_function_space(solver)
    dof_coordinates = np.asarray(V.tabulate_dof_coordinates(), dtype=float)[:, :3]
    positions = np.asarray(getattr(electrodes, "positions", None), dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("electrodes.positions must have shape (n_electrodes, 3)")
    if positions.shape[0] == 0:
        raise ValueError("electrodes must contain at least one position")
    if dof_coordinates.ndim != 2 or dof_coordinates.shape[1] != 3 or dof_coordinates.shape[0] == 0:
        raise ValueError("solver.V.tabulate_dof_coordinates() must return non-empty 3D coordinates")

    nearest_dof_ids, nearest_distances = _nearest_dofs_for_points(dof_coordinates, positions)
    nearest_dof_coordinates = dof_coordinates[nearest_dof_ids]
    labels = list(getattr(electrodes, "labels", [f"E{i + 1}" for i in range(positions.shape[0])]))
    if len(labels) != positions.shape[0]:
        labels = [f"E{i + 1}" for i in range(positions.shape[0])]

    unique_dofs = np.unique(nearest_dof_ids)
    if tol is None:
        indices_exceeding_tol = np.array([], dtype=np.int64)
    else:
        indices_exceeding_tol = np.flatnonzero(nearest_distances > float(tol)).astype(np.int64)

    return {
        "num_electrodes": int(positions.shape[0]),
        "nearest_dof_ids": nearest_dof_ids,
        "nearest_distances": nearest_distances,
        "nearest_dof_coordinates": nearest_dof_coordinates,
        "max_distance": float(np.max(nearest_distances)),
        "mean_distance": float(np.mean(nearest_distances)),
        "num_unique_dofs": int(unique_dofs.size),
        "num_collisions": int(positions.shape[0] - unique_dofs.size),
        "labels": labels,
        "tol": None if tol is None else float(tol),
        "indices_exceeding_tol": indices_exceeding_tol,
    }


def create_electrode_marker_function(
    solver,
    electrodes,
    value_mode: str = "index",
    name: str = "electrodes",
    tol: float | None = None,
):
    """Create a P1 nodal diagnostic marker for electrode locations.

    Each electrode is mapped to the nearest DOLFINx dof. In ``"index"`` mode
    marker values are ``1, 2, ...``; in ``"binary"`` mode every electrode uses
    value ``1``. Collisions are resolved by keeping the largest marker value.
    """
    if value_mode not in {"index", "binary"}:
        raise ValueError("value_mode must be 'index' or 'binary'")

    V = _solver_function_space(solver)
    try:
        from dolfinx import fem
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("creating electrode marker functions requires dolfinx") from exc

    marker = fem.Function(V)
    marker.name = str(name)
    marker.x.array[:] = 0.0

    info = inspect_electrode_marker_mapping(solver, electrodes, tol=tol)
    for electrode_index, dof_id in enumerate(info["nearest_dof_ids"]):
        value = float(electrode_index + 1) if value_mode == "index" else 1.0
        dof_id = int(dof_id)
        marker.x.array[dof_id] = max(float(marker.x.array[dof_id]), value)
    marker.x.scatter_forward()
    return marker


def export_potential_to_xdmf(potential, path, name: str = "potential", time: float = 0.0) -> Path:
    """Export a DOLFINx Function potential to XDMF for ParaView."""
    mesh = _function_mesh(potential)
    io = _require_dolfinx_io()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        potential.name = str(name)
    except AttributeError as exc:
        raise TypeError("potential must be a writable dolfinx.fem.Function") from exc

    with io.XDMFFile(mesh.comm, str(output_path), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(potential, float(time))
    return output_path


def export_dolfinx_function_to_vtx(
    function,
    path,
    name: str = "field",
    time: float = 0.0,
    engine: str = "BP4",
) -> Path:
    """Export a DOLFINx Function to VTX/BP for ParaView."""
    mesh = _function_mesh(function)
    io = _require_dolfinx_io()
    if not hasattr(io, "VTXWriter"):
        raise ImportError("dolfinx.io.VTXWriter is not available; install DOLFINx with ADIOS2 support")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        function.name = str(name)
    except AttributeError as exc:
        raise TypeError("function must be a writable dolfinx.fem.Function") from exc

    with io.VTXWriter(mesh.comm, str(output_path), [function], engine=engine) as vtx:
        vtx.write(float(time))
    return output_path


def export_potential_to_vtx(
    potential,
    path,
    name: str = "potential",
    time: float = 0.0,
    engine: str = "BP4",
) -> Path:
    """Export a DOLFINx Function potential to VTX/BP for ParaView."""
    return export_dolfinx_function_to_vtx(potential, path, name=name, time=time, engine=engine)


def export_forward_result_to_xdmf(
    result: ForwardResult,
    path,
    name: str = "potential",
    time: float = 0.0,
) -> Path:
    """Write result potential plus mesh to XDMF/HDF5 and return the XDMF path.

    Open the ``.xdmf`` file in ParaView and keep its companion ``.h5`` nearby.
    """
    return export_potential_to_xdmf(result.potential, path, name=name, time=time)


def export_forward_result_to_vtx(
    result: ForwardResult,
    path,
    name: str = "potential",
    time: float = 0.0,
    engine: str = "BP4",
) -> Path:
    """Write result potential to VTX/BP and return the output path.

    VTX/BP is the preferred ParaView fallback when XDMF is unstable.
    """
    return export_potential_to_vtx(result.potential, path, name=name, time=time, engine=engine)


def export_electrode_markers_to_vtx(
    solver,
    electrodes,
    path,
    value_mode: str = "index",
    name: str = "electrodes",
    time: float = 0.0,
    engine: str = "BP4",
) -> Path:
    """Export nearest-DOF electrode markers to VTX/BP for ParaView."""
    marker = create_electrode_marker_function(
        solver=solver,
        electrodes=electrodes,
        value_mode=value_mode,
        name=name,
    )
    return export_dolfinx_function_to_vtx(marker, path, name=name, time=time, engine=engine)
