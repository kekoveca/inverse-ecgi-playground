import os

import numpy as np
import pytest

from green import GreenTransferMatrix
from inverse import (
    CandidateInverseSolution,
    SingleDipoleInverseResult,
    SingleDipoleInverseSolver,
    format_inverse_summary,
    inverse_reconstruction_metrics,
    localization_error,
    moment_angle_error_deg,
    moment_relative_error,
    relative_residual,
    residual_norm,
    residual_vector,
    solve_single_dipole_inverse,
    solve_tikhonov_moment,
    summarize_inverse_result,
)


os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("OMPI_MCA_orte_tmpdir_base", "/tmp")

DOLFINX_TESTS_ENABLED = os.environ.get("RUN_DOLFINX_TESTS") == "1"
requires_dolfinx = pytest.mark.skipif(
    not DOLFINX_TESTS_ENABLED,
    reason="set RUN_DOLFINX_TESTS=1 to run real DOLFINx inverse tests",
)


def test_solve_tikhonov_exact():
    A = np.eye(3)
    g = np.array([1.0, 2.0, 3.0])

    assert np.allclose(solve_tikhonov_moment(A, g), g)


def test_solve_tikhonov_overdetermined():
    A = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    true_moment = np.array([0.5, -2.0, 4.0])
    g = A @ true_moment

    assert np.allclose(solve_tikhonov_moment(A, g), true_moment)


def test_solve_tikhonov_regularized():
    A = np.eye(3)
    g = np.array([1.0, 2.0, 3.0])

    assert np.allclose(solve_tikhonov_moment(A, g, lambda_reg=1.0), 0.5 * g)


def test_residual_functions():
    A = np.eye(3)
    p = np.array([1.0, 1.0, 1.0])
    g = np.array([1.0, 2.0, 1.0])

    assert np.allclose(residual_vector(A, p, g), [0.0, -1.0, 0.0])
    assert residual_norm(A, p, g) == pytest.approx(1.0)
    assert relative_residual(A, p, g) == pytest.approx(1.0 / np.linalg.norm(g))


def _synthetic_transfer(sign: float = 1.0) -> GreenTransferMatrix:
    A = np.zeros((3, 4, 3), dtype=float)
    A[0] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    A[1] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    A[2] = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    return GreenTransferMatrix(
        A=A,
        candidate_points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        candidate_cell_ids=np.array([10, 11, 12]),
        sign=sign,
    )


def test_transfer_inverse_recovers_candidate():
    transfer = _synthetic_transfer()
    true_index = 1
    true_moment = np.array([1.0, -2.0, 0.5])
    g = transfer.matrix_for_candidate(true_index) @ true_moment

    result = solve_single_dipole_inverse(transfer, g)

    assert result.best_candidate_index == true_index
    assert np.allclose(result.estimated_moment, true_moment)
    assert result.estimated_cell_id == 11
    assert result.relative_residual < 1e-14


def test_transfer_sign_is_used():
    transfer = _synthetic_transfer(sign=-1.0)
    true_index = 1
    true_moment = np.array([1.0, -2.0, 0.5])
    g = transfer.matrix_for_candidate(true_index) @ true_moment

    result = solve_single_dipole_inverse(transfer, g)

    assert result.best_candidate_index == true_index
    assert np.allclose(result.estimated_moment, true_moment)


def test_metrics():
    assert localization_error([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert moment_angle_error_deg([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(90.0)
    assert moment_relative_error([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)

    transfer = _synthetic_transfer()
    true_moment = np.array([1.0, -2.0, 0.5])
    g = transfer.matrix_for_candidate(1) @ true_moment
    result = solve_single_dipole_inverse(transfer, g)
    metrics = inverse_reconstruction_metrics(
        result,
        true_position=transfer.candidate_points[1],
        true_moment=true_moment,
        localization_threshold=1e-12,
    )

    assert metrics["localization_error"] == pytest.approx(0.0)
    assert metrics["success"] is True


def test_result_summary_omits_full_candidate_arrays():
    transfer = _synthetic_transfer()
    true_moment = np.array([1.0, -2.0, 0.5])
    result = solve_single_dipole_inverse(transfer, transfer.matrix_for_candidate(1) @ true_moment)

    summary = result.to_summary_dict()

    assert result.residual_map().shape == (3,)
    assert result.moment_map().shape == (3, 3)
    assert summary["best_candidate_index"] == 1
    assert summary["num_candidates"] == 3
    assert "candidates" not in summary
    assert "observed_measurements" not in summary


def test_diagnostics_top_k_sorted_by_residual():
    candidates = [
        CandidateInverseSolution(i, [float(i), 0.0, 0.0], i, [1.0, 0.0, 0.0], residual, residual)
        for i, residual in enumerate([3.0, 1.0, 2.0])
    ]
    result = SingleDipoleInverseResult(
        observed_measurements=np.array([1.0, 2.0]),
        best=candidates[1],
        candidates=candidates,
        lambda_reg=0.0,
    )

    summary = summarize_inverse_result(result, top_k=2)
    text = format_inverse_summary(result, top_k=2)

    assert [row["candidate_index"] for row in summary["top_candidates"]] == [1, 2]
    assert "best candidate: 1" in text


@requires_dolfinx
def test_forward_green_inverse_consistency_small_mesh():
    pytest.importorskip("dolfinx")
    pytest.importorskip("mpi4py")
    pytest.importorskip("petsc4py")

    from fem import NeumannPoissonSolver
    from forward import ForwardSolver
    from geometry import ElectrodeSet
    from green import GreenSolver, build_green_transfer_matrix
    from sources import PointDipole
    from verification import create_unit_cube_meshdata

    mesh = create_unit_cube_meshdata(2)
    solver = NeumannPoissonSolver(mesh, pc_type="none", test_nullspace=True)
    try:
        solver.ksp.setTolerances(rtol=1e-13, atol=1e-14, max_it=10000)
        electrodes = ElectrodeSet(
            positions=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
            labels=[f"E{i}" for i in range(8)],
        )
        forward = ForwardSolver(solver, electrodes=electrodes, reference="average")
        candidate_mesh_cell_ids = np.array([2, 17, 35], dtype=np.int64)
        candidate_points = mesh.cell_centers(candidate_mesh_cell_ids)
        true_index = 1
        true_moment = np.array([0.5, -1.0, 2.0])

        green_basis = GreenSolver(solver, forward.measurement_operator).solve_all()
        transfer = build_green_transfer_matrix(
            solver,
            green_basis,
            candidate_points=candidate_points,
        )

        source = PointDipole(position=candidate_points[true_index], moment=true_moment)
        forward_result = forward.solve(source)
        inverse_result = SingleDipoleInverseSolver(
            transfer,
            lambda_reg=1e-12,
            reference="average",
        ).solve(forward_result.measurements)
        metrics = inverse_reconstruction_metrics(
            inverse_result,
            true_position=candidate_points[true_index],
            true_moment=true_moment,
            localization_threshold=1e-12,
        )

        assert inverse_result.best_candidate_index == true_index
        assert metrics["localization_error"] < 1e-12
        assert metrics["moment_angle_error_deg"] < 1e-2
        assert inverse_result.relative_residual < 1e-6
    finally:
        solver.destroy()
