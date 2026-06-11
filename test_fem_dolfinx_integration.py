import os

import numpy as np
import pytest

from geometry import MeshData, read_gmsh_meshio
from fem import FEMProblem, create_dolfinx_mesh
from sources import PointDipole, assemble_point_dipole_rhs_petsc


os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("OMPI_MCA_orte_tmpdir_base", "/tmp")

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DOLFINX_TESTS") != "1",
    reason="set RUN_DOLFINX_TESTS=1 to run real DOLFINx integration tests:\n\
        RUN_DOLFINX_TESTS=1 pytest -v test_fem_dolfinx_integration.py",
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


def test_real_dolfinx_stack_is_importable():
    dolfinx = pytest.importorskip("dolfinx")
    pytest.importorskip("basix")
    pytest.importorskip("ufl")
    pytest.importorskip("mpi4py")
    pytest.importorskip("petsc4py")

    assert dolfinx.__version__


def test_create_dolfinx_mesh_builds_real_dolfinx_mesh():
    mesh = single_tetra_mesh()

    domain = create_dolfinx_mesh(mesh)

    assert domain.topology.dim == 3
    assert domain.geometry.dim == 3
    assert domain.topology.index_map(domain.topology.dim).size_local == mesh.num_cells


def test_fem_problem_assembles_stiffness_matrix_and_solves_repeated_rhs():
    problem = FEMProblem(single_tetra_mesh(), pc_type="none", test_nullspace=True)
    K = problem.K

    try:
        assert K is not None
        assert problem.diagnostics.nullspace_test_passed is True

        rhs1 = problem.rhs_from_local_array(np.array([1.0, -1.0, 0.0, 0.0]))
        u1 = problem.solve(rhs1)

        rhs2 = problem.rhs_from_local_array(np.array([0.0, 1.0, -1.0, 0.0]))
        u2 = problem.solve(rhs2)

        assert problem.K is K
        assert abs(float(u1.x.array.mean())) < 1e-12
        assert abs(float(u2.x.array.mean())) < 1e-12
        assert problem.diagnostics.converged_reason is not None
        assert problem.diagnostics.converged_reason > 0
        assert problem.diagnostics.residual_norm is not None
        assert problem.diagnostics.residual_norm < 1e-8
    finally:
        problem.destroy()


def test_point_dipole_petsc_rhs_solves_with_real_fem_problem():
    problem = FEMProblem(single_tetra_mesh(), pc_type="none", test_nullspace=True)
    source = PointDipole(position=[0.25, 0.25, 0.25], moment=[1.0, 2.0, 3.0])

    try:
        rhs = assemble_point_dipole_rhs_petsc(problem, source)
        u = problem.solve(rhs)

        assert abs(float(rhs.x.array.sum())) < 1e-12
        assert abs(float(u.x.array.mean())) < 1e-12
        assert problem.diagnostics.converged_reason is not None
        assert problem.diagnostics.converged_reason > 0
    finally:
        problem.destroy()


def test_real_gmsh_meshdata_block_can_become_dolfinx_mesh():
    mesh = read_gmsh_meshio("torso.msh", dim=3)
    volume_mesh = mesh.to_mesh_data("tetra", physical_name="domain")

    domain = create_dolfinx_mesh(volume_mesh)

    assert volume_mesh.num_cells == 47158
    assert volume_mesh.metadata["physical_dimension"] == 3
    assert volume_mesh.metadata["physical_tag"] == 1
    assert domain.topology.dim == 3
    assert domain.geometry.dim == 3
