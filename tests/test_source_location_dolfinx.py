import os

import numpy as np
import pytest

from geometry import MeshData
from sources import PointDipole


os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("OMPI_MCA_orte_tmpdir_base", "/tmp")

DOLFINX_TESTS_ENABLED = os.environ.get("RUN_DOLFINX_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not DOLFINX_TESTS_ENABLED,
    reason="set RUN_DOLFINX_TESTS=1 to run real DOLFINx source location tests",
)


def single_tetra_mesh():
    return MeshData(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
        cell_type="tetra",
    )


def two_tetra_mesh():
    first = single_tetra_mesh().points
    points = np.vstack([first, first + np.array([2.0, 0.0, 0.0])])
    return MeshData(
        points=points,
        cells=np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64),
        cell_type="tetra",
    )


def make_solver(mesh):
    pytest.importorskip("dolfinx")
    pytest.importorskip("mpi4py")
    pytest.importorskip("petsc4py")

    from fem import NeumannPoissonSolver

    return NeumannPoissonSolver(mesh, pc_type="none", test_nullspace=True)


def test_locate_point_in_dolfinx_p1_tetra_mesh_one_tetra():
    from sources import locate_point_in_dolfinx_p1_tetra_mesh

    solver = make_solver(single_tetra_mesh())
    try:
        assert locate_point_in_dolfinx_p1_tetra_mesh(solver, [0.25, 0.25, 0.25]) == 0
    finally:
        solver.destroy()


def test_barycentric_coordinates_are_inside_used_dolfinx_cell():
    from sources import inspect_point_dipole_location_petsc

    solver = make_solver(single_tetra_mesh())
    source = PointDipole(position=[0.25, 0.25, 0.25], moment=[0.0, 0.0, 1.0])
    try:
        info = inspect_point_dipole_location_petsc(solver, source)

        assert info["is_inside_used_dolfinx_cell"] is True
        assert info["barycentric_min"] >= -1e-10
        assert info["barycentric_sum"] == pytest.approx(1.0)
        assert np.allclose(info["barycentric_in_dolfinx_cell"], [0.25, 0.25, 0.25, 0.25])
        assert info["is_near_cell_boundary"] is False
        assert info["cell_boundary_kind"] == "interior"
        assert info["location_ambiguity_warning"] is None
    finally:
        solver.destroy()


def test_petsc_assembly_does_not_trust_source_cell_id_by_default():
    from sources import (
        assemble_point_dipole_rhs_petsc,
        get_nonzero_dofs_from_rhs,
        inspect_point_dipole_location_petsc,
        locate_point_in_dolfinx_p1_tetra_mesh,
    )

    solver = make_solver(two_tetra_mesh())
    position = np.array([2.25, 0.25, 0.25])
    try:
        expected_cell_id = locate_point_in_dolfinx_p1_tetra_mesh(solver, position)
        wrong_cell_id = 1 - expected_cell_id
        source = PointDipole(position=position, moment=[1.0, 2.0, 3.0]).with_cell_id(wrong_cell_id)

        info = inspect_point_dipole_location_petsc(solver, source, trust_source_cell_id=False)
        rhs = assemble_point_dipole_rhs_petsc(solver, source, trust_source_cell_id=False)
        expected_dofs = np.asarray(solver.V.dofmap.cell_dofs(expected_cell_id), dtype=np.int64)

        assert info["declared_cell_id"] == wrong_cell_id
        assert info["used_cell_id"] == expected_cell_id
        assert info["used_cell_id"] != source.cell_id
        assert set(get_nonzero_dofs_from_rhs(rhs).tolist()) == set(expected_dofs.tolist())
    finally:
        solver.destroy()


def test_trust_source_cell_id_uses_declared_id():
    from sources import inspect_point_dipole_location_petsc, locate_point_in_dolfinx_p1_tetra_mesh

    solver = make_solver(two_tetra_mesh())
    position = np.array([2.25, 0.25, 0.25])
    try:
        located_cell_id = locate_point_in_dolfinx_p1_tetra_mesh(solver, position)
        declared_cell_id = 1 - located_cell_id
        source = PointDipole(position=position, moment=[1.0, 2.0, 3.0]).with_cell_id(declared_cell_id)

        info = inspect_point_dipole_location_petsc(solver, source, trust_source_cell_id=True)

        assert info["used_cell_id"] == declared_cell_id
        assert info["is_inside_used_dolfinx_cell"] is False
    finally:
        solver.destroy()


def test_meshdata_and_dolfinx_cell_centers_match_for_one_tetra():
    from sources import compare_meshdata_and_dolfinx_cell_centers

    solver = make_solver(single_tetra_mesh())
    try:
        comparison = compare_meshdata_and_dolfinx_cell_centers(solver)

        assert comparison["num_checked"] == 1
        assert comparison["worst_cell_id"] == 0
        assert comparison["max_diff"] < 1e-12
        assert comparison["mean_diff"] < 1e-12
    finally:
        solver.destroy()


def test_create_cell_marker_function_marks_only_cell_dofs():
    from sources import create_cell_marker_function, get_nonzero_dofs_from_rhs

    solver = make_solver(single_tetra_mesh())
    try:
        marker = create_cell_marker_function(solver, cell_id=0)
        cell_dofs = np.asarray(solver.V.dofmap.cell_dofs(0), dtype=np.int64)

        assert marker.name == "source_marker"
        assert set(get_nonzero_dofs_from_rhs(marker).tolist()) == set(cell_dofs.tolist())
        assert np.allclose(marker.x.array[cell_dofs], 1.0)
    finally:
        solver.destroy()
