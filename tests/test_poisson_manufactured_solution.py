import os

import numpy as np
import pytest

from verification import build_convergence_report, create_unit_cube_meshdata, u_exact_neumann_cosine, format_convergence_report


os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("OMPI_MCA_orte_tmpdir_base", "/tmp")

DOLFINX_TESTS_ENABLED = os.environ.get("RUN_DOLFINX_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not DOLFINX_TESTS_ENABLED,
    reason="set RUN_DOLFINX_TESTS=1 to run manufactured-solution convergence tests",
)


def require_dolfinx():
    pytest.importorskip("dolfinx")
    pytest.importorskip("mpi4py")
    pytest.importorskip("petsc4py")


def solve_manufactured_error(n: int) -> float:
    require_dolfinx()
    from fem import NeumannPoissonSolver

    solver = NeumannPoissonSolver(
        create_unit_cube_meshdata(n),
        degree=1,
        sigma=1.0,
        ksp_type="cg",
        pc_type="hypre",
        test_nullspace=True,
    )
    try:
        if solver.comm.size != 1:
            pytest.skip("manufactured convergence test currently supports serial execution")

        x = solver.ufl.SpatialCoordinate(solver.domain)
        exact_ufl = (
            solver.ufl.cos(2.0 * np.pi * x[0])
            * solver.ufl.cos(2.0 * np.pi * x[1])
            * solver.ufl.cos(2.0 * np.pi * x[2])
        )
        forcing = 12.0 * np.pi**2 * exact_ufl
        test_function = solver.ufl.TestFunction(solver.V)
        linear_form = solver.fem.form(forcing * test_function * solver.ufl.dx(domain=solver.domain))
        rhs = solver.fem_petsc.assemble_vector(linear_form)

        numerical = solver.solve(rhs)
        exact = solver.fem.Function(solver.V)
        exact.interpolate(lambda coordinates: u_exact_neumann_cosine(coordinates[:3].T))
        exact.x.array[:] -= exact.x.array.mean()
        exact.x.scatter_forward()

        difference = numerical - exact
        error_form = solver.fem.form(
            solver.ufl.inner(difference, difference) * solver.ufl.dx(domain=solver.domain)
        )
        error_squared = float(solver.fem.assemble_scalar(error_form))
        return float(np.sqrt(max(error_squared, 0.0)))
    finally:
        solver.destroy()


def test_neumann_poisson_manufactured_convergence():
    levels = (4, 8, 16, 32, 64, 128)
    h_values = [1.0 / n for n in levels]
    errors = [solve_manufactured_error(n) for n in levels]
    report = build_convergence_report(h_values, errors)
    print(format_convergence_report(report))

    assert report.errors_decrease
    assert report.min_rate is not None
    assert report.min_rate > 1.0
