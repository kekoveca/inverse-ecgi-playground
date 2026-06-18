import os

import numpy as np
import pytest

from sources import PointDipole
from verification import create_unit_cube_meshdata


os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("OMPI_MCA_orte_tmpdir_base", "/tmp")

DOLFINX_TESTS_ENABLED = os.environ.get("RUN_DOLFINX_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not DOLFINX_TESTS_ENABLED,
    reason="set RUN_DOLFINX_TESTS=1 to run forward convergence tests",
)


def require_dolfinx():
    pytest.importorskip("dolfinx")
    pytest.importorskip("mpi4py")
    pytest.importorskip("petsc4py")


def observation_points_face_centers() -> np.ndarray:
    return np.array(
        [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 1.0],
            [0.5, 0.0, 0.5],
            [0.5, 1.0, 0.5],
            [0.0, 0.5, 0.5],
            [1.0, 0.5, 0.5],
        ]
    )


def build_dolfinx_dof_measurement_operator(solver, points):
    """Build a test-only point operator in the solver's actual DOF ordering."""
    from scipy.sparse import csr_matrix

    from measurements import MeasurementOperator, reference_matrix

    points = np.asarray(points, dtype=float)
    dof_coords = np.asarray(solver.V.tabulate_dof_coordinates(), dtype=float)[:, :3]
    dof_ids = []
    for point in points:
        distances = np.linalg.norm(dof_coords - point, axis=1)
        dof_id = int(np.argmin(distances))
        if distances[dof_id] > 1e-12:
            raise ValueError(f"observation point {point.tolist()} is not a mesh dof")
        dof_ids.append(dof_id)

    rows = np.arange(points.shape[0], dtype=np.int64)
    interpolation = csr_matrix(
        (np.ones(points.shape[0]), (rows, np.asarray(dof_ids, dtype=np.int64))),
        shape=(points.shape[0], dof_coords.shape[0]),
    )
    reference = reference_matrix(points.shape[0], reference="average", sparse=True)
    return MeasurementOperator(
        interpolation_matrix=interpolation,
        reference_matrix=reference,
        electrode_cell_ids=np.zeros(points.shape[0], dtype=np.int64),
        electrode_barycentric=np.zeros((points.shape[0], 4), dtype=float),
        reference="average",
        labels=[f"E{i}" for i in range(points.shape[0])],
        metadata={"ordering": "dolfinx_dof", "purpose": "verification"},
    )


def make_forward_solver(n: int):
    require_dolfinx()
    from fem import NeumannPoissonSolver
    from forward import ForwardSolver

    solver = NeumannPoissonSolver(
        create_unit_cube_meshdata(n),
        degree=1,
        sigma=1.0,
        ksp_type="cg",
        pc_type="hypre",
        test_nullspace=True,
    )
    measurement_operator = build_dolfinx_dof_measurement_operator(solver, observation_points_face_centers())
    return solver, ForwardSolver(
        poisson_solver=solver,
        measurement_operator=measurement_operator,
        reference="average",
    )


def test_forward_solver_sanity_and_rhs_localization():
    from sources import assemble_point_dipole_rhs_petsc, get_nonzero_dofs_from_rhs, inspect_point_dipole_location_petsc

    solver, forward = make_forward_solver(4)
    source = PointDipole(position=[0.5, 0.5, 0.5], moment=[1.0, np.sqrt(2.0), np.pi])
    try:
        info = inspect_point_dipole_location_petsc(solver, source)
        rhs = assemble_point_dipole_rhs_petsc(solver, source)
        result = forward.solve(source)

        assert set(get_nonzero_dofs_from_rhs(rhs).tolist()) == set(info["cell_dofs"].tolist())
        assert len(get_nonzero_dofs_from_rhs(rhs)) == 4
        assert float(rhs.x.array.sum()) == pytest.approx(0.0, abs=1e-10)
        assert np.all(np.isfinite(result.nodal_values))
        assert np.all(np.isfinite(result.measurements))
        assert float(result.measurements.sum()) == pytest.approx(0.0, abs=1e-10)
    finally:
        solver.destroy()


def test_forward_repeated_solve_is_deterministic():
    solver, forward = make_forward_solver(4)
    source = PointDipole(position=[0.5, 0.5, 0.5], moment=[0.0, 0.0, 1.0])
    try:
        first = forward.solve(source).measurements
        second = forward.solve(source).measurements

        assert np.allclose(second, first, rtol=1e-8, atol=1e-10)
    finally:
        solver.destroy()


def test_forward_linearity_in_moment():
    solver, forward = make_forward_solver(4)
    position = [0.5, 0.5, 0.5]
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    try:
        g1 = forward.solve(PointDipole(position=position, moment=p1)).measurements
        g2 = forward.solve(PointDipole(position=position, moment=p2)).measurements
        g12 = forward.solve(PointDipole(position=position, moment=p1 + p2)).measurements

        assert np.allclose(g12, g1 + g2, rtol=5e-6, atol=1e-7)
    finally:
        solver.destroy()


def test_forward_scales_with_moment_amplitude():
    solver, forward = make_forward_solver(4)
    position = [0.5, 0.5, 0.5]
    moment = np.array([0.0, 0.0, 1.0])
    try:
        g1 = forward.solve(PointDipole(position=position, moment=moment)).measurements
        g2 = forward.solve(PointDipole(position=position, moment=2.0 * moment)).measurements

        assert np.allclose(g2, 2.0 * g1, rtol=1e-6, atol=1e-8)
    finally:
        solver.destroy()


def test_forward_measurements_stabilize_under_refinement():
    measurements = []
    # Avoid grid vertices/faces: assigning a singular dipole to one of several
    # incident cells is ordering-dependent and is not a refinement sequence.
    source_position = [0.47, 0.53, 0.51]
    for n in (4, 8, 16):
        solver, forward = make_forward_solver(n)
        try:
            result = forward.solve(PointDipole(position=source_position, moment=[0.0, 0.0, 1.0]))
            assert np.all(np.isfinite(result.measurements))
            assert float(result.measurements.sum()) == pytest.approx(0.0, abs=1e-8)
            measurements.append(result.measurements.copy())
        finally:
            solver.destroy()

    diff_4_8 = np.linalg.norm(measurements[1] - measurements[0])
    diff_8_16 = np.linalg.norm(measurements[2] - measurements[1])

    assert diff_8_16 < diff_4_8
