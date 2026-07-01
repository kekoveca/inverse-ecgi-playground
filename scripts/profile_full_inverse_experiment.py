#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
for path in (PROJECT_ROOT, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import full_inverse_experiment_torso as example
from fem import NeumannPoissonSolver
from forward import extract_nodal_values, export_dolfinx_function_to_vtx
from green import (
    GreenSolver,
    build_green_transfer_matrix,
    check_measurement_matrix_compatibility,
    measurement_matrix_row_sums,
)
from inverse import SingleDipoleInverseSolver
from measurements import build_measurement_operator
from performance import (
    PerformanceTimer,
    format_timing_table,
    get_process_memory_mb,
    save_timing_csv,
    save_timing_json,
)
from sources import PointDipole, assemble_point_dipole_rhs_petsc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the full forward -> Green -> inverse pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mesh", default="torso_refined.msh")
    parser.add_argument("--output", default="output/performance_profile")
    parser.add_argument("--domain-name", default="domain")
    parser.add_argument("--boundary-name", default="boundary")
    parser.add_argument("--num-electrodes", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--moment", nargs=3, type=float, default=[0.0, 0.0, 1.0])
    parser.add_argument("--reference", choices=["average", "single", "none"], default="average")
    parser.add_argument("--reference-index", type=int, default=None)
    parser.add_argument("--lambda-reg", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--electrode-mode", choices=["surface-farthest", "surface-random"], default="surface-farthest")
    parser.add_argument("--electrode-offset", type=float, default=0.0)
    parser.add_argument("--source-index", type=int, default=None)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--skip-green", action="store_true")
    parser.add_argument("--skip-inverse", action="store_true")
    parser.add_argument("--max-green-rows", type=int, default=None)
    return parser.parse_args()


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


def memory_snapshot(stage: str, snapshots: list[dict[str, Any]], **metadata) -> None:
    snapshots.append(
        {
            "stage": stage,
            "memory_mb": get_process_memory_mb(),
            **metadata,
        }
    )


def write_summary(path: Path, *, timer: PerformanceTimer, memory_snapshots: list[dict], metadata: dict) -> None:
    sorted_records = sorted(timer.records, key=lambda record: record.elapsed_s, reverse=True)
    lines = [
        "# Performance Profile Summary",
        "",
        "## Run metadata",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Timing",
            "",
            format_timing_table(timer),
            "",
            "## Top stages",
            "",
        ]
    )
    for record in sorted_records[:8]:
        lines.append(f"- `{record.name}`: {record.elapsed_s:.6g} s")
    lines.extend(["", "## Memory snapshots", "", "| Stage | Memory, MB | Metadata |", "| --- | ---: | --- |"])
    for snapshot in memory_snapshots:
        memory = snapshot.get("memory_mb")
        memory_text = "" if memory is None else f"{memory:.3f}"
        metadata_text = ", ".join(
            f"{key}={value}" for key, value in snapshot.items() if key not in {"stage", "memory_mb"}
        )
        lines.append(f"| `{snapshot['stage']}` | {memory_text} | {metadata_text} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    example.require_dolfinx_runtime()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timer = PerformanceTimer()
    memory: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "mesh": args.mesh,
        "num_electrodes": args.num_electrodes,
        "num_candidates": args.num_candidates,
        "reference": args.reference,
        "reference_index": args.reference_index,
        "max_green_rows": args.max_green_rows,
    }

    solver = None
    with timer.time("total"):
        mesh_path = Path(args.mesh)
        with timer.time("read_mesh", mesh=str(mesh_path)):
            tagged = example.read_gmsh_meshio(mesh_path, dim=3)
        memory_snapshot("after_read_mesh", memory)

        with timer.time("convert_volume_mesh", physical_name=args.domain_name):
            volume_mesh = tagged.to_mesh_data(cell_type="tetra", physical_name=args.domain_name)
        with timer.time("convert_surface_mesh", physical_name=args.boundary_name):
            surface_mesh = tagged.to_mesh_data(cell_type="triangle", physical_name=args.boundary_name)
        mesh_summary = example.mesh_diagnostics_summary(volume_mesh, surface_mesh)
        metadata.update(mesh_summary)
        memory_snapshot("after_meshdata", memory, **mesh_summary)

        with timer.time("build_electrodes", num_electrodes=args.num_electrodes):
            electrodes, electrode_projection, electrode_surface = example.build_demo_surface_electrodes(
                volume_mesh=volume_mesh,
                surface_mesh=surface_mesh,
                num_electrodes=args.num_electrodes,
                mode=args.electrode_mode,
                seed=args.seed,
                offset=args.electrode_offset,
            )
        metadata["electrode_num_projected"] = electrode_projection["num_projected"]
        metadata["electrode_nearest_surface_distance_max"] = electrode_surface["max_nearest_surface_distance"]
        memory_snapshot("after_electrodes", memory, num_electrodes=electrodes.num_electrodes)

        with timer.time("build_source_region", num_candidates=args.num_candidates):
            candidate_points, _, source_region = example.build_source_candidates(
                volume_mesh,
                num_candidates=args.num_candidates,
                seed=args.seed,
            )
        true_index = example.choose_true_source_index(candidate_points, args.source_index, volume_mesh)
        source = PointDipole(position=candidate_points[true_index], moment=np.asarray(args.moment, dtype=float))
        metadata["selected_candidates"] = int(candidate_points.shape[0])
        metadata["source_region_name"] = source_region.name
        memory_snapshot("after_candidates", memory, num_candidates=int(candidate_points.shape[0]))

        with timer.time("create_solver", num_cells=volume_mesh.num_cells, num_points=volume_mesh.num_points):
            solver = NeumannPoissonSolver(mesh=volume_mesh, degree=1, sigma=args.sigma)
        memory_snapshot("after_solver_created", memory)

        try:
            with timer.time("build_measurement_operator", num_electrodes=electrodes.num_electrodes):
                measurement_operator = build_measurement_operator(
                    mesh=volume_mesh,
                    electrodes=electrodes,
                    reference=args.reference,
                    reference_index=args.reference_index,
                    surface_mesh=surface_mesh,
                )
            memory_snapshot("after_measurement_operator", memory)

            with timer.time("forward_rhs"):
                rhs = assemble_point_dipole_rhs_petsc(solver, source)
            with timer.time("forward_solve"):
                potential = solver.solve(rhs)
            with timer.time("forward_measurements"):
                dof_values = extract_nodal_values(potential)
                node_to_dof = None
                if measurement_operator.metadata.get("ordering", "meshdata_node") == "meshdata_node":
                    from fem import build_node_to_dof_map_p1

                    node_to_dof = build_node_to_dof_map_p1(solver)
                    measurement_values = dof_values[node_to_dof]
                else:
                    measurement_values = dof_values
                raw_measurements = measurement_operator.evaluate_raw(measurement_values)
                measurements = measurement_operator.evaluate(measurement_values)
            metadata["raw_measurement_norm"] = float(np.linalg.norm(raw_measurements))
            metadata["measurement_norm"] = float(np.linalg.norm(measurements))
            memory_snapshot("after_forward", memory)

            with timer.time("check_green_rhs_compatibility"):
                row_sums = measurement_matrix_row_sums(measurement_operator)
                green_compatible = check_measurement_matrix_compatibility(measurement_operator)
            metadata["green_rhs_max_abs_row_sum"] = float(np.max(np.abs(row_sums))) if row_sums.size else 0.0
            metadata["green_compatible"] = bool(green_compatible)

            transfer = None
            if not args.skip_green:
                if not green_compatible:
                    raise ValueError("measurement matrix rows are not compatible with pure Neumann Green solves")
                if args.max_green_rows is None:
                    row_indices = np.arange(measurement_operator.num_electrodes, dtype=np.int64)
                else:
                    row_indices = np.arange(
                        min(int(args.max_green_rows), measurement_operator.num_electrodes),
                        dtype=np.int64,
                    )
                green_solver = GreenSolver(solver, measurement_operator, keep_functions=True)
                with timer.time("green_solve_all", num_rows=int(row_indices.size)):
                    green_basis = green_solver.solve_all(row_indices=row_indices)
                timer.add_record(
                    "green_solve_per_row_mean",
                    timer.records[-1].elapsed_s / max(1, int(row_indices.size)),
                    num_rows=int(row_indices.size),
                )
                memory_snapshot("after_green_basis", memory, num_green_functions=len(green_basis.functions))

                with timer.time(
                    "build_transfer_matrix",
                    num_candidates=int(candidate_points.shape[0]),
                    num_measurements=len(green_basis.functions),
                ):
                    transfer = build_green_transfer_matrix(solver, green_basis, candidate_points=candidate_points)
                metadata["transfer_shape"] = tuple(int(x) for x in transfer.A.shape)
                memory_snapshot("after_transfer_matrix", memory, transfer_shape=metadata["transfer_shape"])

                if not args.skip_inverse:
                    inverse_measurements = measurements[transfer.measurement_row_indices]
                    with timer.time(
                        "inverse_solve",
                        num_candidates=transfer.num_candidates,
                        num_measurements=transfer.num_measurements,
                    ):
                        inverse_result = SingleDipoleInverseSolver(
                            transfer,
                            lambda_reg=args.lambda_reg,
                            reference=args.reference,
                        ).solve(inverse_measurements)
                    metadata["inverse_best_candidate"] = int(inverse_result.best_candidate_index)
                    metadata["inverse_relative_residual"] = float(inverse_result.relative_residual)
                    memory_snapshot("after_inverse", memory)

            if not args.no_export:
                with timer.time("exports"):
                    export_dolfinx_function_to_vtx(potential, output_dir / "potential.bp", name="potential")
        finally:
            solver.destroy()

    save_timing_csv(timer, output_dir / "timing.csv")
    save_timing_json(timer, output_dir / "timing.json")
    write_json(output_dir / "memory.json", {"snapshots": memory})
    write_summary(output_dir / "profile_summary.md", timer=timer, memory_snapshots=memory, metadata=metadata)
    print(format_timing_table(timer))
    print(f"Saved performance profile to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
