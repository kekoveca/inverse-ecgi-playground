import os

import numpy as np
import pytest

from geometry import MeshData
from sources import PointDipole, gradients_p1_tetra


os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("OMPI_MCA_orte_tmpdir_base", "/tmp")

DOLFINX_TESTS_ENABLED = os.environ.get("RUN_DOLFINX_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not DOLFINX_TESTS_ENABLED,
    reason="set RUN_DOLFINX_TESTS=1 to run real DOLFINx source RHS tests",
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


def make_solver():
    pytest.importorskip("dolfinx")
    pytest.importorskip("mpi4py")
    pytest.importorskip("petsc4py")

    from fem import NeumannPoissonSolver

    return NeumannPoissonSolver(single_tetra_mesh(), pc_type="none", test_nullspace=True)


def test_petsc_rhs_nonzero_dofs_are_cell_dofs():
    from sources import assemble_point_dipole_rhs_petsc, get_nonzero_dofs_from_rhs

    solver = make_solver()
    source = PointDipole(position=[0.25, 0.25, 0.25], moment=[1.0, 2.0, 3.0])
    try:
        rhs = assemble_point_dipole_rhs_petsc(solver, source)
        cell_dofs = np.asarray(solver.V.dofmap.cell_dofs(0), dtype=np.int64)
        nonzero_dofs = get_nonzero_dofs_from_rhs(rhs)

        assert set(nonzero_dofs.tolist()) == set(cell_dofs.tolist())
        assert len(nonzero_dofs) == 4
        assert float(rhs.x.array.sum()) == pytest.approx(0.0, abs=1e-12)
    finally:
        solver.destroy()


def test_petsc_rhs_local_values_match_gradients_in_dof_order():
    from sources import assemble_point_dipole_rhs_petsc

    solver = make_solver()
    source = PointDipole(position=[0.25, 0.25, 0.25], moment=[1.0, 2.0, 3.0])
    try:
        rhs = assemble_point_dipole_rhs_petsc(solver, source)
        cell_dofs = np.asarray(solver.V.dofmap.cell_dofs(0), dtype=np.int64)
        dof_coords = np.asarray(solver.V.tabulate_dof_coordinates(), dtype=float)
        vertices = dof_coords[cell_dofs, :3]
        expected_local_rhs = gradients_p1_tetra(vertices) @ source.moment

        assert np.allclose(rhs.x.array[cell_dofs], expected_local_rhs)
    finally:
        solver.destroy()


def test_petsc_rhs_respects_source_cell_id():
    from sources import assemble_point_dipole_rhs_petsc, inspect_point_dipole_rhs_petsc

    solver = make_solver()
    source = PointDipole(position=[0.25, 0.25, 0.25], moment=[1.0, 2.0, 3.0]).with_cell_id(0)
    try:
        rhs = assemble_point_dipole_rhs_petsc(solver, source)
        info = inspect_point_dipole_rhs_petsc(solver, source)

        assert info["cell_id"] == 0
        assert set(info["nonzero_dofs"].tolist()) == set(solver.V.dofmap.cell_dofs(0).tolist())
        assert np.allclose(rhs.x.array[info["cell_dofs"]], info["local_rhs"])
        assert info["local_rhs_sum"] == pytest.approx(0.0, abs=1e-12)
    finally:
        solver.destroy()
