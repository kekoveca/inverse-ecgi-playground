import os
from types import SimpleNamespace

import numpy as np
import pytest

from green import (
    GreenSolver,
    GreenTransferMatrix,
    build_green_transfer_matrix,
    check_measurement_matrix_compatibility,
    compare_forward_and_green,
    extract_measurement_rhs_row,
    infer_green_sign_from_cases,
    load_green_transfer_matrix,
    measurement_matrix_row_sums,
    save_green_transfer_matrix,
)


os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("OMPI_MCA_orte_tmpdir_base", "/tmp")

DOLFINX_TESTS_ENABLED = os.environ.get("RUN_DOLFINX_TESTS") == "1"
requires_dolfinx = pytest.mark.skipif(
    not DOLFINX_TESTS_ENABLED,
    reason="set RUN_DOLFINX_TESTS=1 to run real DOLFINx Green tests",
)


class FakeMeasurementOperator:
    def __init__(self, matrix):
        self._matrix = matrix

    def matrix(self):
        return self._matrix


def test_measurement_matrix_compatibility_dense():
    operator = FakeMeasurementOperator(
        np.array(
            [
                [1.0, -1.0, 0.0],
                [0.5, 0.5, -1.0],
            ]
        )
    )

    assert np.allclose(measurement_matrix_row_sums(operator), [0.0, 0.0])
    assert check_measurement_matrix_compatibility(operator)
    assert np.allclose(extract_measurement_rhs_row(operator, 1), [0.5, 0.5, -1.0])


def test_measurement_matrix_incompatible():
    operator = FakeMeasurementOperator(np.array([[1.0, 0.0, 0.0]]))

    assert not check_measurement_matrix_compatibility(operator)


def test_transfer_matrix_predict_respects_sign():
    A = np.arange(18, dtype=float).reshape(2, 3, 3)
    moment = np.array([1.0, -2.0, 0.5])
    transfer = GreenTransferMatrix(
        A=A,
        candidate_points=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        candidate_cell_ids=np.array([4, 7]),
        sign=-1.0,
    )

    assert np.allclose(transfer.matrix_for_candidate(0), -A[0])
    assert np.allclose(transfer.predict(0, moment), -(A[0] @ moment))


def test_transfer_matrix_records_measurement_row_indices():
    transfer = GreenTransferMatrix(
        A=np.ones((1, 3, 3), dtype=float),
        candidate_points=np.array([[0.0, 0.0, 0.0]]),
        candidate_cell_ids=np.array([0]),
        measurement_row_indices=np.array([2, 4, 7]),
    )

    assert np.array_equal(transfer.measurement_row_indices, [2, 4, 7])
    assert transfer.to_summary_dict()["measurement_row_indices"] == [2, 4, 7]
    assert transfer.metadata["measurement_row_indices"] == [2, 4, 7]


def test_compare_forward_and_green_finds_sign():
    A = np.zeros((1, 2, 3), dtype=float)
    A[0, :, 0] = [2.0, -3.0]
    transfer = GreenTransferMatrix(
        A=A,
        candidate_points=np.array([[0.2, 0.2, 0.2]]),
        candidate_cell_ids=np.array([0]),
    )
    forward_result = SimpleNamespace(measurements=np.array([-2.0, 3.0]))

    diagnostics = compare_forward_and_green(forward_result, transfer, 0, [1.0, 0.0, 0.0])

    assert diagnostics["best_sign"] == -1.0
    assert diagnostics["best_rel_error"] == pytest.approx(0.0)
    assert infer_green_sign_from_cases([diagnostics, diagnostics]) == -1.0


def test_green_transfer_cache_roundtrip(tmp_path):
    transfer = GreenTransferMatrix(
        A=np.arange(18, dtype=float).reshape(2, 3, 3),
        candidate_points=np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]]),
        candidate_cell_ids=np.array([2, 5]),
        sign=-1.0,
        metadata={"reference": "average", "version": 1},
        measurement_row_indices=np.array([3, 5, 8]),
    )

    path = save_green_transfer_matrix(transfer, tmp_path / "green_transfer.npz")
    loaded = load_green_transfer_matrix(path)

    assert path.exists()
    assert np.allclose(loaded.A, transfer.A)
    assert np.allclose(loaded.candidate_points, transfer.candidate_points)
    assert np.array_equal(loaded.candidate_cell_ids, transfer.candidate_cell_ids)
    assert np.array_equal(loaded.measurement_row_indices, transfer.measurement_row_indices)
    assert loaded.sign == -1.0
    assert loaded.metadata == transfer.metadata


def _single_tetra_mesh():
    from geometry import MeshData

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


@requires_dolfinx
def test_node_to_dof_map_p1_one_tetra():
    from green import build_node_to_dof_map_p1

    solver = _make_solver()
    try:
        node_to_dof = build_node_to_dof_map_p1(solver)
        mapping = solver.p1_node_dof_mapping()
        dof_coords = solver.V.tabulate_dof_coordinates()

        assert sorted(node_to_dof.tolist()) == [0, 1, 2, 3]
        assert np.array_equal(mapping.node_to_dof, node_to_dof)
        assert np.array_equal(mapping.dof_to_node[node_to_dof], np.arange(4))
        assert solver.p1_node_dof_mapping() is mapping
        assert np.allclose(dof_coords[node_to_dof, :3], solver.mesh_data.points)
    finally:
        solver.destroy()


@requires_dolfinx
def test_create_green_rhs_function_row_sum():
    from geometry import ElectrodeSet
    from green import create_green_rhs_function
    from measurements import build_measurement_operator

    solver = _make_solver()
    try:
        electrodes = ElectrodeSet(
            positions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            labels=["E1", "E2"],
        )
        operator = build_measurement_operator(solver.mesh_data, electrodes, reference="average")
        rhs = create_green_rhs_function(solver, operator, row_index=0)

        assert float(rhs.x.array.sum()) == pytest.approx(0.0, abs=1e-12)
    finally:
        solver.destroy()


@requires_dolfinx
def test_gradient_on_dolfinx_cell_linear_function():
    from green import gradient_on_dolfinx_cell

    solver = _make_solver()
    try:
        function = solver.zero_function()
        coordinates = solver.V.tabulate_dof_coordinates()
        function.x.array[:] = 2.0 * coordinates[:, 0] - 3.0 * coordinates[:, 1] + 4.0 * coordinates[:, 2] + 5.0
        function.x.scatter_forward()

        gradient = gradient_on_dolfinx_cell(solver, function, cell_id=0)

        assert np.allclose(gradient, [2.0, -3.0, 4.0], atol=1e-12)
    finally:
        solver.destroy()


@requires_dolfinx
def test_green_forward_consistency_small_mesh():
    from forward import ForwardSolver
    from geometry import ElectrodeSet
    from sources import PointDipole

    solver = _make_solver()
    try:
        electrodes = ElectrodeSet(
            positions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            labels=["E1", "E2", "E3"],
        )
        forward = ForwardSolver(solver, electrodes=electrodes, reference="average")
        candidate = np.array([0.25, 0.25, 0.25])
        source = PointDipole(position=candidate, moment=[1.0, 2.0, 3.0])
        forward_result = forward.solve(source)

        green_basis = GreenSolver(solver, forward.measurement_operator).solve_all()
        transfer = build_green_transfer_matrix(
            solver,
            green_basis,
            candidate_points=np.asarray([candidate]),
        )
        diagnostics = compare_forward_and_green(
            forward_result,
            transfer,
            candidate_index=0,
            moment=source.moment,
        )

        assert diagnostics["rel_error_plus"] < 1e-6
        assert diagnostics["best_sign"] == 1.0
        assert np.all(np.isfinite(transfer.A))
        assert np.array_equal(transfer.measurement_row_indices, [0, 1, 2])
        assert transfer.metadata["num_boundary_candidates"] == 0
    finally:
        solver.destroy()
