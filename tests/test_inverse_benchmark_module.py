import csv
import os
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark import (
    ForwardBenchmarkRecord,
    ForwardBenchmarkResult,
    InverseBenchmarkRecord,
    InverseBenchmarkResult,
    InverseBenchmarkScenario,
    NoNoise,
    ForwardBenchmarkRunner,
    ForwardBenchmarkScenario,
    filter_forward_result_by_electrode_set,
    make_all_electrodes_subset,
    run_inverse_benchmark,
    save_inverse_benchmark_result,
)
from green import GreenTransferMatrix


os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("OMPI_MCA_orte_tmpdir_base", "/tmp")

DOLFINX_TESTS_ENABLED = os.environ.get("RUN_DOLFINX_TESTS") == "1"
requires_dolfinx = pytest.mark.skipif(
    not DOLFINX_TESTS_ENABLED,
    reason="set RUN_DOLFINX_TESTS=1 to run inverse benchmark DOLFINx integration",
)


def synthetic_transfer() -> GreenTransferMatrix:
    A = np.zeros((3, 5, 3), dtype=float)
    A[0] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    A[1] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, -1.0, 0.5],
        ]
    )
    A[2] = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
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
        candidate_cell_ids=np.array([0, 1, 2], dtype=np.int64),
    )


def make_forward_record(
    transfer: GreenTransferMatrix,
    true_index: int = 1,
    p_true: np.ndarray | None = None,
    electrode_set_name: str = "all",
    noise_model_name: str = "none",
) -> ForwardBenchmarkRecord:
    if p_true is None:
        p_true = np.array([1.0, -2.0, 0.5])
    clean = transfer.matrix_for_candidate(true_index) @ p_true
    noise = np.zeros_like(clean)
    noisy = clean + noise
    return ForwardBenchmarkRecord(
        scenario_name="forward_unit",
        source_index=true_index,
        source_position=transfer.candidate_points[true_index],
        source_moment=p_true,
        source_cell_id=int(transfer.candidate_cell_ids[true_index]),
        electrode_set_name=electrode_set_name,
        num_electrodes=clean.size,
        noise_model_name=noise_model_name,
        reference="average",
        clean_measurements=clean,
        noisy_measurements=noisy,
        noise=noise,
    )


def make_forward_result(records) -> ForwardBenchmarkResult:
    return ForwardBenchmarkResult(
        scenario={"name": "forward_unit", "num_sources": len(records)},
        records=list(records),
    )


def test_inverse_benchmark_from_forward_result_clean():
    transfer = synthetic_transfer()
    record = make_forward_record(transfer)
    forward_result = make_forward_result([record])

    result = run_inverse_benchmark(
        forward_result,
        transfer,
        name="inverse_unit",
        use_clean_measurements=True,
        use_noisy_measurements=False,
        localization_threshold=1e-12,
    )

    assert len(result.records) == 1
    inverse_record = result.records[0]
    assert inverse_record.measurement_kind == "clean"
    assert inverse_record.estimated_candidate_index == 1
    assert inverse_record.localization_error == pytest.approx(0.0)
    assert inverse_record.moment_relative_error < 1e-12
    assert inverse_record.moment_angle_error_deg < 1e-8
    assert inverse_record.success is True


def test_inverse_benchmark_clean_and_noisy():
    transfer = synthetic_transfer()
    record = make_forward_record(transfer, noise_model_name="synthetic")
    noisy = record.clean_measurements + np.array([0.01, -0.02, 0.0, 0.01, -0.01])
    record = ForwardBenchmarkRecord(
        scenario_name=record.scenario_name,
        source_index=record.source_index,
        source_position=record.source_position,
        source_moment=record.source_moment,
        source_cell_id=record.source_cell_id,
        electrode_set_name=record.electrode_set_name,
        num_electrodes=record.num_electrodes,
        noise_model_name=record.noise_model_name,
        reference=record.reference,
        clean_measurements=record.clean_measurements,
        noisy_measurements=noisy,
        noise=noisy - record.clean_measurements,
    )

    result = run_inverse_benchmark(make_forward_result([record]), transfer)

    assert len(result.records) == 2
    assert {record.measurement_kind for record in result.records} == {"clean", "noisy"}


def test_inverse_benchmark_summary_statistics():
    records = [
        InverseBenchmarkRecord(
            scenario_name="inverse",
            source_index=index,
            source_position=[0.0, 0.0, 0.0],
            source_moment=[1.0, 0.0, 0.0],
            source_cell_id=index,
            electrode_set_name="all",
            num_electrodes=5,
            noise_model_name="none",
            measurement_kind="clean",
            lambda_reg=0.0,
            estimated_candidate_index=index,
            estimated_position=[float(error), 0.0, 0.0],
            estimated_cell_id=index,
            estimated_moment=[1.0, 0.0, 0.0],
            residual_norm=0.1 * index,
            relative_residual=0.01 * index,
            localization_error=error,
            moment_relative_error=0.0,
            moment_angle_error_deg=float(index),
            success=error <= 1.0,
        )
        for index, error in enumerate([0.0, 1.0, 3.0])
    ]
    result = InverseBenchmarkResult(
        scenario={"name": "inverse", "lambda_reg": 0.0},
        records=records,
    )

    summary = result.summary()

    assert summary["num_records"] == 3
    assert summary["localization_error_mean"] == pytest.approx(4.0 / 3.0)
    assert summary["localization_error_median"] == pytest.approx(1.0)
    assert summary["localization_error_p90"] == pytest.approx(2.6)
    assert summary["success_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["relative_residual_mean"] == pytest.approx(0.01)


def test_inverse_scenario_rejects_mismatched_measurement_length():
    transfer = synthetic_transfer()
    bad_record = make_forward_record(transfer)
    bad_record = ForwardBenchmarkRecord(
        scenario_name=bad_record.scenario_name,
        source_index=bad_record.source_index,
        source_position=bad_record.source_position,
        source_moment=bad_record.source_moment,
        source_cell_id=bad_record.source_cell_id,
        electrode_set_name=bad_record.electrode_set_name,
        num_electrodes=3,
        noise_model_name=bad_record.noise_model_name,
        reference=bad_record.reference,
        clean_measurements=np.ones(3),
        noisy_measurements=np.ones(3),
        noise=np.zeros(3),
    )
    scenario = InverseBenchmarkScenario(
        name="bad",
        forward_result=make_forward_result([bad_record]),
        transfer_matrix=transfer,
    )

    with pytest.raises(ValueError, match="measurement length"):
        scenario.validate()


def test_filter_forward_result_by_electrode_set():
    transfer = synthetic_transfer()
    first = make_forward_record(transfer, electrode_set_name="all")
    second = make_forward_record(transfer, electrode_set_name="subset")
    result = make_forward_result([first, second])

    filtered = filter_forward_result_by_electrode_set(result, "subset")

    assert len(filtered.records) == 1
    assert filtered.records[0].electrode_set_name == "subset"
    assert filtered.metadata["filtered_electrode_set_name"] == "subset"


def test_save_inverse_benchmark_result(tmp_path):
    transfer = synthetic_transfer()
    result = run_inverse_benchmark(
        make_forward_result([make_forward_record(transfer)]),
        transfer,
        use_noisy_measurements=False,
    )

    output = save_inverse_benchmark_result(result, tmp_path / "inverse")

    assert (output / "inverse_config.json").exists()
    assert (output / "inverse_records.csv").exists()
    assert (output / "inverse_summary.json").exists()
    with (output / "inverse_records.csv").open("r", encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert "estimated_candidate_index" in row
    assert "localization_error" in row


@requires_dolfinx
def test_end_to_end_inverse_benchmark_small_mesh():
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
    candidate_mesh_cell_ids = np.array([2, 17, 35], dtype=np.int64)
    candidate_points = mesh.cell_centers(candidate_mesh_cell_ids)
    true_index = 1
    true_moment = np.array([0.5, -1.0, 2.0])

    solver = NeumannPoissonSolver(mesh, pc_type="none", test_nullspace=True)
    try:
        solver.ksp.setTolerances(rtol=1e-13, atol=1e-14, max_it=10000)
        forward = ForwardSolver(solver, electrodes=electrodes, reference="average")
        green_basis = GreenSolver(solver, forward.measurement_operator).solve_all()
        transfer = build_green_transfer_matrix(solver, green_basis, candidate_points=candidate_points)
    finally:
        solver.destroy()

    sources = [PointDipole(position=candidate_points[true_index], moment=true_moment)]
    forward_scenario = ForwardBenchmarkScenario(
        name="forward_inverse_smoke",
        geometry=SimpleNamespace(volume_mesh=mesh),
        sources=sources,
        electrode_sets=[make_all_electrodes_subset(electrodes)],
        noise_models=[NoNoise()],
        reference="average",
    )
    def solver_factory(volume_mesh):
        problem = NeumannPoissonSolver(volume_mesh, pc_type="none", test_nullspace=True)
        problem.ksp.setTolerances(rtol=1e-13, atol=1e-14, max_it=10000)
        return problem

    forward_result = ForwardBenchmarkRunner(poisson_solver_factory=solver_factory).run(forward_scenario)

    inverse_result = run_inverse_benchmark(
        forward_result,
        transfer,
        lambda_reg=1e-12,
        localization_threshold=1e-12,
        use_clean_measurements=True,
        use_noisy_measurements=False,
    )

    assert len(inverse_result.records) == 1
    record = inverse_result.records[0]
    assert record.estimated_candidate_index == true_index
    assert record.localization_error < 1e-12
    assert record.relative_residual < 1e-6
