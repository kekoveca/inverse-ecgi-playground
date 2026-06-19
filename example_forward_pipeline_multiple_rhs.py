from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fem import NeumannPoissonSolver
from forward import (
    ForwardSolver,
    export_forward_result_to_vtx,
)
from geometry import (
    read_gmsh_meshio,
    TorsoGeometry,
    SourceRegion,
    ElectrodeSet,
)
from sources import (
    PointDipole,
    inspect_point_dipole_location_petsc,
    locate_point_in_mesh,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the Neumann Poisson forward problem on torso.msh and export potential to ParaView."
    )
    parser.add_argument("--mesh", default="torso.msh", help="Path to the Gmsh .msh file.")
    parser.add_argument("--physical-name", default="domain", help="Gmsh physical volume name for tetra cells.")
    parser.add_argument(
        "--bbox",
        nargs=6,
        type=float,
        default=(-20.0, -10.0, -20.0, 20.0, 10.0, 20.0),
        metavar=("Xmin", "Ymin", "Zmin", "Xmax", "Ymax", "Zmax"),
        help="Sources bbox.",
    )
    parser.add_argument(
        "--moment",
        nargs=3,
        type=float,
        default=(1.0, 0.0, 0.0),
        metavar=("PX", "PY", "PZ"),
        help="Point dipole moment.",
    )
    parser.add_argument("--sigma", type=float, default=1.0, help="Constant scalar conductivity.")
    parser.add_argument("--ksp-type", default="cg", choices=("cg", "gmres", "preonly"), help="PETSc KSP type.")
    parser.add_argument(
        "--pc-type",
        default="hypre",
        choices=("hypre", "gamg", "jacobi", "lu", "none"),
        help="PETSc preconditioner type.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path = Path(args.mesh)

    tagged_mesh = read_gmsh_meshio(mesh_path, dim=3)
    volume_mesh = tagged_mesh.to_mesh_data(cell_type="tetra", physical_name=args.physical_name)

    solver = NeumannPoissonSolver(
        mesh=volume_mesh,
        degree=1,
        sigma=args.sigma,
        ksp_type=args.ksp_type,
        pc_type=args.pc_type,
    )

    sources_bbox_bounds = args.bbox
    source_region = SourceRegion.from_bounding_box(volume_mesh, sources_bbox_bounds[0:3], sources_bbox_bounds[3:6])

    electrodes = ElectrodeSet(np.array([[30.0, 0.0, 0.0], [-30.0, 0.0, 0.0]]))

    torso = TorsoGeometry("test", volume_mesh, electrodes, source_region)

    moment = np.asarray(args.moment, dtype=float)

    forward = ForwardSolver(poisson_solver=solver, reference="average")

    try:
        for position, cell_id in zip(source_region.candidate_points, source_region.candidate_cell_ids):
            source = PointDipole(position=position, moment=moment).with_cell_id(cell_id)

            print("=" * 12)
            print(f"Loaded mesh: {mesh_path}")
            print(f"Physical volume: {args.physical_name!r}")
            print(f"Volume mesh: {volume_mesh.num_points} nodes, {volume_mesh.num_cells} tetrahedra")
            print(f"Dipole position: {source.position.tolist()}")
            print(f"Dipole moment: {source.moment.tolist()}")
            print(f"Dipole MeshData cell_id: {source.cell_id}")

            location_info = inspect_point_dipole_location_petsc(solver, source)

            result = forward.solve(source)
            vtx_path = export_forward_result_to_vtx(result, f"output/cell_{cell_id}/potential.bp")

            summary = result.to_dict()
            summary.update(
                {
                    "mesh": str(mesh_path),
                    "physical_name": args.physical_name,
                    "num_cells": volume_mesh.num_cells,
                    "vtx_path": str(vtx_path),
                    "source_location": {
                        "meshdata_cell_id": location_info["meshdata_located_cell_id"],
                        "dolfinx_cell_id": location_info["used_cell_id"],
                        "cell_dofs": location_info["cell_dofs"].tolist(),
                        "dof_cell_center": location_info["dof_cell_center"].tolist(),
                        "barycentric": location_info["barycentric_in_dolfinx_cell"].tolist(),
                        "inside": location_info["is_inside_used_dolfinx_cell"],
                        "ordering_warning": location_info["ordering_warning"],
                    },
                    "solver_diagnostics": {
                        "ksp_type": solver.diagnostics.ksp_type,
                        "pc_type": solver.diagnostics.pc_type,
                        "converged_reason": solver.diagnostics.converged_reason,
                        "residual_norm": solver.diagnostics.residual_norm,
                        "nullspace_test_passed": solver.diagnostics.nullspace_test_passed,
                    },
                }
            )

            summary_path = Path(f"output/cell_{cell_id}/forward_summary.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            print(f"VTX/BP potential exported to: {vtx_path}")
            print(f"DOLFINx source cell_id: {location_info['used_cell_id']}")
            print(f"Source barycentric coordinates: {location_info['barycentric_in_dolfinx_cell']}")
            print(f"Summary written to: {summary_path}")
            print(f"PETSc converged_reason: {solver.diagnostics.converged_reason}")
            print(f"PETSc residual_norm: {solver.diagnostics.residual_norm}")
    finally:
        solver.destroy()


if __name__ == "__main__":
    main()
