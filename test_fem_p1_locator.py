import os

import numpy as np
import pytest

from geometry import MeshData


os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("OMPI_MCA_orte_tmpdir_base", "/tmp")

DOLFINX_TESTS_ENABLED = os.environ.get("RUN_DOLFINX_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not DOLFINX_TESTS_ENABLED,
    reason="set RUN_DOLFINX_TESTS=1 to run real DOLFINx P1 locator tests",
)


def _single_tetra_mesh():
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


def _make_solver():
    pytest.importorskip("dolfinx")
    pytest.importorskip("mpi4py")
    pytest.importorskip("petsc4py")

    from fem import NeumannPoissonSolver

    return NeumannPoissonSolver(_single_tetra_mesh(), pc_type="none", test_nullspace=True)


def test_p1_tetra_locator_locates_one_tetra_point():
    solver = _make_solver()
    try:
        locator = solver.p1_tetra_locator()
        point = np.array([0.25, 0.25, 0.25])

        cell_id = locator.locate_point(point)
        cell_ids, barycentric = locator.locate_points([point], return_barycentric=True)

        assert cell_id == 0
        assert np.array_equal(cell_ids, [0])
        assert np.allclose(barycentric.sum(axis=1), [1.0])
        assert np.all(barycentric >= -1e-12)
        assert solver.p1_tetra_locator() is locator
    finally:
        solver.destroy()


def test_p1_tetra_locator_cell_geometry_matches_dof_order():
    solver = _make_solver()
    try:
        locator = solver.p1_tetra_locator()
        cell_dofs, vertices = locator.cell_geometry([0])
        dof_coords = solver.V.tabulate_dof_coordinates()

        assert cell_dofs.shape == (1, 4)
        assert vertices.shape == (1, 4, 3)
        assert np.allclose(vertices[0], dof_coords[cell_dofs[0], :3])
        assert locator.basis_gradients([0]).shape == (1, 4, 3)
    finally:
        solver.destroy()


def test_source_rhs_uses_cached_locator():
    from sources import PointDipole, assemble_point_dipole_rhs_petsc, get_nonzero_dofs_from_rhs

    solver = _make_solver()
    try:
        locator = solver.p1_tetra_locator()
        source = PointDipole(position=[0.25, 0.25, 0.25], moment=[1.0, 2.0, 3.0])
        rhs = assemble_point_dipole_rhs_petsc(solver, source)

        assert solver.p1_tetra_locator() is locator
        assert set(get_nonzero_dofs_from_rhs(rhs).tolist()) == set(locator.cell_dofs[0].tolist())
    finally:
        solver.destroy()
