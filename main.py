from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fem import NeumannPoissonSolver
from forward import ForwardSolver, export_dolfinx_function_to_vtx, export_forward_result_to_vtx, export_forward_result_to_xdmf
from geometry import read_gmsh_meshio
from sources import (
    PointDipole,
    assemble_point_dipole_rhs_petsc,
    create_cell_marker_function,
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
        "--position",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Point dipole position.",
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
    parser.add_argument("--output", default="output/potential.xdmf", help="Output XDMF path for ParaView.")
    parser.add_argument("--vtx-output", default="output/potential.bp", help="Output VTX/BP path for ParaView.")
    parser.add_argument("--rhs-output", default="output/rhs.bp", help="Diagnostic RHS VTX/BP path for ParaView.")
    parser.add_argument(
        "--marker-output",
        default="output/source_marker.bp",
        help="Diagnostic source-cell marker VTX/BP path for ParaView.",
    )
    parser.add_argument(
        "--summary",
        default="output/forward_summary.json",
        help="Path for a small JSON summary of the run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path = Path(args.mesh)
    output_path = Path(args.output)
    vtx_output_path = Path(args.vtx_output)
    rhs_output_path = Path(args.rhs_output)
    marker_output_path = Path(args.marker_output)
    summary_path = Path(args.summary)

    tagged_mesh = read_gmsh_meshio(mesh_path, dim=3)
    volume_mesh = tagged_mesh.to_mesh_data(cell_type="tetra", physical_name=args.physical_name)

    position = np.asarray(args.position, dtype=float)
    moment = np.asarray(args.moment, dtype=float)
    cell_id = locate_point_in_mesh(volume_mesh, position)
    source = PointDipole(position=position, moment=moment).with_cell_id(cell_id)

    print(f"Loaded mesh: {mesh_path}")
    print(f"Physical volume: {args.physical_name!r}")
    print(f"Volume mesh: {volume_mesh.num_points} nodes, {volume_mesh.num_cells} tetrahedra")
    print(f"Dipole position: {source.position.tolist()}")
    print(f"Dipole moment: {source.moment.tolist()}")
    print(f"Dipole MeshData cell_id: {source.cell_id}")

    solver = NeumannPoissonSolver(
        mesh=volume_mesh,
        degree=1,
        sigma=args.sigma,
        ksp_type=args.ksp_type,
        pc_type=args.pc_type,
    )
    try:
        location_info = inspect_point_dipole_location_petsc(solver, source)
        marker = create_cell_marker_function(solver, location_info["used_cell_id"])
        marker_path = export_dolfinx_function_to_vtx(marker, marker_output_path, name="source_marker")

        rhs = assemble_point_dipole_rhs_petsc(solver, source)
        rhs_path = export_dolfinx_function_to_vtx(rhs, rhs_output_path, name="rhs")

        forward = ForwardSolver(poisson_solver=solver, reference="average")
        result = forward.solve(source)
        xdmf_path = export_forward_result_to_xdmf(result, output_path)
        vtx_path = export_forward_result_to_vtx(result, vtx_output_path)

        summary = result.to_dict()
        summary.update(
            {
                "mesh": str(mesh_path),
                "physical_name": args.physical_name,
                "num_cells": volume_mesh.num_cells,
                "xdmf_path": str(xdmf_path),
                "vtx_path": str(vtx_path),
                "rhs_path": str(rhs_path),
                "source_marker_path": str(marker_path),
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
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"XDMF potential exported to: {xdmf_path}")
        print(f"VTX/BP potential exported to: {vtx_path}")
        print(f"VTX/BP RHS exported to: {rhs_path}")
        print(f"VTX/BP source marker exported to: {marker_path}")
        print(f"DOLFINx source cell_id: {location_info['used_cell_id']}")
        print(f"Source barycentric coordinates: {location_info['barycentric_in_dolfinx_cell']}")
        print("Open the .xdmf file in ParaView. If ParaView crashes or shows an empty file, open the .bp output.")
        print(f"Summary written to: {summary_path}")
        print(f"PETSc converged_reason: {solver.diagnostics.converged_reason}")
        print(f"PETSc residual_norm: {solver.diagnostics.residual_norm}")
    finally:
        solver.destroy()


if __name__ == "__main__":
    main()
