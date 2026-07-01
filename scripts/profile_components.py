#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from green import GreenBasis, GreenTransferMatrix, build_green_transfer_matrix
from inverse import SingleDipoleInverseSolver
from performance import PerformanceTimer, format_timing_table, save_timing_csv, save_timing_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile isolated performance-sensitive components.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--component",
        choices=["inverse-scaling", "point-location", "green-transfer"],
        default="inverse-scaling",
    )
    parser.add_argument("--output", default="output/component_profile")
    parser.add_argument("--candidate-counts", nargs="+", type=int, default=[100, 1000, 10000])
    parser.add_argument("--measurement-counts", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--mesh", default=None, help="Optional .msh mesh for DOLFINx component profiling.")
    parser.add_argument("--domain-name", default="domain")
    parser.add_argument("--unit-cube-n", type=int, default=8)
    parser.add_argument("--num-location-points", type=int, default=100)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--num-measurements", type=int, default=16)
    parser.add_argument("--lambda-reg", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def profile_inverse_scaling(args: argparse.Namespace) -> PerformanceTimer:
    timer = PerformanceTimer()
    rng = np.random.default_rng(args.seed)
    p_true = np.array([0.7, -1.2, 0.5])
    for num_measurements in args.measurement_counts:
        for num_candidates in args.candidate_counts:
            A = rng.normal(size=(num_candidates, num_measurements, 3))
            true_index = min(num_candidates // 3, num_candidates - 1)
            candidate_points = rng.normal(size=(num_candidates, 3))
            candidate_cell_ids = np.arange(num_candidates, dtype=np.int64)
            transfer = GreenTransferMatrix(
                A=A,
                candidate_points=candidate_points,
                candidate_cell_ids=candidate_cell_ids,
            )
            measurements = transfer.matrix_for_candidate(true_index) @ p_true
            solver = SingleDipoleInverseSolver(transfer, lambda_reg=args.lambda_reg)
            with timer.time(
                "inverse_solve",
                num_candidates=int(num_candidates),
                num_measurements=int(num_measurements),
            ):
                result = solver.solve(measurements)
            timer.records[-1].metadata["best_candidate"] = int(result.best_candidate_index)
            timer.records[-1].metadata["relative_residual"] = float(result.relative_residual)
    return timer


def _require_dolfinx_profile_runtime():
    missing = []
    for module_name in ("dolfinx", "mpi4py", "petsc4py"):
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise ImportError(
            "DOLFINx component profiling requires installed runtime modules: " + ", ".join(missing)
        )


def _load_profile_mesh(args: argparse.Namespace):
    if args.mesh is None:
        from verification import create_unit_cube_meshdata

        return create_unit_cube_meshdata(args.unit_cube_n)

    from geometry import read_gmsh_meshio

    tagged = read_gmsh_meshio(args.mesh, dim=3)
    return tagged.to_mesh_data(cell_type="tetra", physical_name=args.domain_name)


def _sample_cell_center_points(locator, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_cells = int(locator.cell_centers.shape[0])
    count = min(int(count), num_cells)
    if count <= 0:
        raise ValueError("count must be positive and mesh must contain local cells")
    cell_ids = rng.choice(num_cells, size=count, replace=False).astype(np.int64)
    return locator.cell_centers[cell_ids].copy(), cell_ids


def profile_point_location(args: argparse.Namespace) -> PerformanceTimer:
    _require_dolfinx_profile_runtime()
    from fem import NeumannPoissonSolver

    timer = PerformanceTimer()
    mesh = _load_profile_mesh(args)
    solver = NeumannPoissonSolver(mesh, pc_type="none", test_nullspace=False)
    try:
        with timer.time("build_p1_tetra_locator", num_cells=mesh.num_cells, num_points=mesh.num_points):
            locator = solver.p1_tetra_locator()
        points, expected_cell_ids = _sample_cell_center_points(locator, args.num_location_points, args.seed)
        with timer.time("locate_points", num_points=int(points.shape[0]), num_cells=mesh.num_cells):
            located = locator.locate_points(points)
        timer.records[-1].metadata["all_expected"] = bool(np.array_equal(located, expected_cell_ids))
    finally:
        solver.destroy()
    return timer


def profile_green_transfer(args: argparse.Namespace) -> PerformanceTimer:
    _require_dolfinx_profile_runtime()
    from fem import NeumannPoissonSolver

    timer = PerformanceTimer()
    rng = np.random.default_rng(args.seed)
    mesh = _load_profile_mesh(args)
    solver = NeumannPoissonSolver(mesh, pc_type="none", test_nullspace=False)
    try:
        with timer.time("build_p1_tetra_locator", num_cells=mesh.num_cells, num_points=mesh.num_points):
            locator = solver.p1_tetra_locator()
        candidate_points, _ = _sample_cell_center_points(locator, args.num_candidates, args.seed)

        functions = []
        for _ in range(int(args.num_measurements)):
            function = solver.zero_function()
            function.x.array[:] = rng.normal(size=function.x.array.shape)
            function.x.scatter_forward()
            functions.append(function)
        measurement_operator = SimpleNamespace(num_electrodes=len(functions), num_nodes=mesh.num_points)
        green_basis = GreenBasis(
            measurement_operator=measurement_operator,
            functions=functions,
            reference="profile",
            metadata={"row_indices": list(range(len(functions)))},
        )

        with timer.time(
            "green_transfer_with_location",
            num_candidates=int(candidate_points.shape[0]),
            num_measurements=len(functions),
        ):
            transfer = build_green_transfer_matrix(solver, green_basis, candidate_points=candidate_points)
        timer.records[-1].metadata["transfer_shape"] = tuple(int(x) for x in transfer.A.shape)

        with timer.time("locate_candidates", num_candidates=int(candidate_points.shape[0])):
            cell_ids = locator.locate_points(candidate_points)
        with timer.time(
            "green_transfer_with_prelocated_cells",
            num_candidates=int(candidate_points.shape[0]),
            num_measurements=len(functions),
        ):
            prelocated = build_green_transfer_matrix(
                solver,
                green_basis,
                candidate_points=candidate_points,
                candidate_cell_ids=cell_ids,
            )
        timer.records[-1].metadata["transfer_shape"] = tuple(int(x) for x in prelocated.A.shape)
    finally:
        solver.destroy()
    return timer


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.component == "inverse-scaling":
        timer = profile_inverse_scaling(args)
    elif args.component == "point-location":
        timer = profile_point_location(args)
    elif args.component == "green-transfer":
        timer = profile_green_transfer(args)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(args.component)

    save_timing_csv(timer, output_dir / "timing.csv")
    save_timing_json(timer, output_dir / "timing.json")
    (output_dir / "profile_summary.md").write_text(
        "# Component Performance Profile\n\n" + format_timing_table(timer) + "\n",
        encoding="utf-8",
    )
    print(format_timing_table(timer))
    print(f"Saved component profile to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
