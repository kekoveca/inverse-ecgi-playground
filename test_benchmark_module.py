import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark import (
    AbsoluteGaussianNoise,
    ForwardBenchmarkRecord,
    ForwardBenchmarkResult,
    ForwardBenchmarkRunner,
    ForwardBenchmarkScenario,
    NoNoise,
    RelativeGaussianNoise,
    axis_moments,
    compute_snr_db,
    correlation,
    forward_signal_metrics,
    generate_random_sources_from_region,
    generate_sources_from_region,
    load_records_csv,
    make_all_electrodes_subset,
    noise_metrics,
    relative_l2_error,
    rmse,
    save_forward_benchmark_result,
    select_electrodes_by_indices,
    select_farthest_point_electrodes,
    select_random_electrodes,
)
from geometry import ElectrodeSet, MeshData, SourceRegion, TorsoGeometry
from sources import PointDipole


os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("OMPI_MCA_orte_tmpdir_base", "/tmp")

DOLFINX_TESTS_ENABLED = os.environ.get("RUN_DOLFINX_TESTS") == "1"
requires_dolfinx = pytest.mark.skipif(
    not DOLFINX_TESTS_ENABLED,
    reason="set RUN_DOLFINX_TESTS=1 to run benchmark DOLFINx integration",
)


def source_region_fixture():
    return SourceRegion(
        candidate_points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        candidate_cell_ids=np.arange(5, dtype=np.int64),
    )


def electrodes_fixture():
    return ElectrodeSet(
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        labels=[f"E{i}" for i in range(6)],
    )


def test_source_set_generators_and_axis_moments():
    region = source_region_fixture()
    moments = axis_moments(scale=2.0)
    source_set = generate_sources_from_region(region, moments=moments)
    limited = generate_sources_from_region(region, moments=moments[:1], max_positions=2)

    assert len(moments) == 3
    assert np.allclose(moments, np.eye(3) * 2.0)
    assert len(source_set) == region.num_candidates * len(moments)
    assert len(source_set.to_table()) == len(source_set)
    assert len(limited) == 2
    assert all(source.cell_id is not None for source in source_set)


def test_random_source_generation_is_reproducible():
    region = source_region_fixture()
    first = generate_random_sources_from_region(region, n_positions=3, moments=[[0.0, 0.0, 1.0]], seed=42)
    second = generate_random_sources_from_region(region, n_positions=3, moments=[[0.0, 0.0, 1.0]], seed=42)

    assert np.allclose([source.position for source in first], [source.position for source in second])
    assert [source.cell_id for source in first] == [source.cell_id for source in second]
    with pytest.raises(ValueError, match="exceeds"):
        generate_random_sources_from_region(region, n_positions=10)


def test_electrode_subset_selection():
    electrodes = electrodes_fixture()
    subset = select_electrodes_by_indices(electrodes, [4, 1, 3], name="chosen")
    random_first = select_random_electrodes(electrodes, n=3, seed=7)
    random_second = select_random_electrodes(electrodes, n=3, seed=7)
    farthest = select_farthest_point_electrodes(electrodes, n=4, start_index=0)
    all_subset = make_all_electrodes_subset(electrodes)

    assert subset.labels == ["E4", "E1", "E3"]
    assert np.array_equal(subset.indices, [4, 1, 3])
    assert np.array_equal(random_first.indices, random_second.indices)
    assert len(farthest) == 4
    assert np.unique(farthest.indices).size == 4
    assert len(all_subset) == electrodes.num_electrodes
    with pytest.raises(ValueError, match="positive"):
        select_random_electrodes(electrodes, n=0)
    with pytest.raises(ValueError, match="exceeds"):
        select_farthest_point_electrodes(electrodes, n=20)


def test_noise_models_are_reproducible_and_do_not_mutate_input():
    values = np.array([1.0, -2.0, 3.0, -4.0])
    original = values.copy()

    no_noise_values, no_noise = NoNoise().apply(values)
    absolute_first, absolute_noise_first = AbsoluteGaussianNoise(0.1, seed=3).apply(values)
    absolute_second, absolute_noise_second = AbsoluteGaussianNoise(0.1, seed=3).apply(values)
    relative_values, relative_noise = RelativeGaussianNoise(30.0, seed=4).apply(values)

    assert np.array_equal(values, original)
    assert np.array_equal(no_noise_values, values)
    assert np.array_equal(no_noise, np.zeros_like(values))
    assert np.allclose(absolute_first, absolute_second)
    assert np.allclose(absolute_noise_first, absolute_noise_second)
    assert compute_snr_db(values, relative_noise) == pytest.approx(30.0)
    assert np.allclose(relative_values, values + relative_noise)


def test_metrics_known_values_and_edge_cases():
    reference = np.array([1.0, 2.0, 3.0])
    estimate = np.array([2.0, 2.0, 2.0])

    assert rmse(reference, estimate) == pytest.approx(np.sqrt(2.0 / 3.0))
    assert relative_l2_error(reference, reference) == pytest.approx(0.0)
    assert correlation(reference, reference) == pytest.approx(1.0)
    assert correlation(reference, -reference) == pytest.approx(-1.0)
    assert np.isnan(correlation(np.ones(3), reference))
    assert compute_snr_db(reference, np.zeros(3)) == np.inf

    forward_metrics = forward_signal_metrics(reference, estimate)
    noise_report = noise_metrics(reference, reference + 0.1, np.full(3, 0.1))
    assert set(forward_metrics) == {"rmse", "relative_l2_error", "max_abs_error", "correlation"}
    assert np.isfinite(noise_report["snr_db"])


def make_record(index: int, length: int, noise_scale: float = 0.1) -> ForwardBenchmarkRecord:
    clean = np.linspace(-1.0, 1.0, length)
    noise = np.full(length, noise_scale)
    noisy = clean + noise
    return ForwardBenchmarkRecord(
        scenario_name="unit",
        source_index=index,
        source_position=np.array([index, 0.0, 0.0]),
        source_moment=np.array([0.0, 0.0, 1.0]),
        source_cell_id=index,
        electrode_set_name=f"set_{length}",
        num_electrodes=length,
        noise_model_name="noise",
        reference="average",
        clean_measurements=clean,
        noisy_measurements=noisy,
        noise=noise,
        metrics=noise_metrics(clean, noisy, noise),
    )


def scenario_fixture():
    source = PointDipole(position=[0.25, 0.25, 0.25], moment=[0.0, 0.0, 1.0])
    return ForwardBenchmarkScenario(
        name="unit",
        geometry=SimpleNamespace(volume_mesh="mesh"),
        sources=[source],
        electrode_sets=[make_all_electrodes_subset(electrodes_fixture())],
        noise_models=[NoNoise()],
        reference="average",
        metadata={"purpose": "test"},
    )


def test_scenario_and_result_summaries_omit_large_arrays():
    scenario = scenario_fixture()
    scenario.validate()
    records = [make_record(0, 2), make_record(0, 4)]
    result = ForwardBenchmarkResult(scenario=scenario, records=records)

    config = scenario.to_config_dict()
    row = records[0].to_row()
    summary = result.summary()

    assert config["num_sources"] == 1
    assert "geometry" not in config
    assert "clean_measurements" not in row
    assert "noisy_measurements" not in row
    assert summary["num_records"] == 2
    assert summary["scenario_name"] == "unit"
    assert summary["electrode_set_names"] == ["set_2", "set_4"]


def test_save_forward_benchmark_result_with_variable_lengths(tmp_path):
    result = ForwardBenchmarkResult(
        scenario=scenario_fixture(),
        records=[make_record(0, 2), make_record(1, 4)],
    )

    output = save_forward_benchmark_result(result, tmp_path / "benchmark")

    assert (output / "config.json").exists()
    assert (output / "records.csv").exists()
    assert (output / "measurements.npz").exists()
    assert (output / "summary.json").exists()
    assert json.loads((output / "summary.json").read_text())["num_records"] == 2
    assert len(load_records_csv(output / "records.csv")) == 2
    with np.load(output / "measurements.npz") as arrays:
        assert {"clean_000000", "noisy_000000", "noise_000000"}.issubset(arrays.files)
        assert arrays["clean_000000"].shape == (2,)
        assert arrays["clean_000001"].shape == (4,)


@requires_dolfinx
def test_forward_benchmark_runner_one_tetra():
    pytest.importorskip("dolfinx")
    pytest.importorskip("mpi4py")
    pytest.importorskip("petsc4py")

    from fem import NeumannPoissonSolver

    mesh = MeshData(
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
    electrodes = ElectrodeSet(
        positions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        labels=["E1", "E2"],
    )
    region = SourceRegion.from_cell_ids(mesh, np.array([0], dtype=np.int64))
    geometry = TorsoGeometry("one_tetra", mesh, electrodes, region)
    scenario = ForwardBenchmarkScenario(
        name="one_tetra",
        geometry=geometry,
        sources=[PointDipole(position=[0.25, 0.25, 0.25], moment=[1.0, 2.0, 3.0])],
        electrode_sets=[make_all_electrodes_subset(electrodes)],
        noise_models=[NoNoise()],
        reference="average",
    )
    runner = ForwardBenchmarkRunner(
        poisson_solver_factory=lambda volume_mesh: NeumannPoissonSolver(
            volume_mesh,
            degree=1,
            sigma=1.0,
            pc_type="none",
        )
    )

    result = runner.run(scenario)

    assert len(result) == 1
    assert np.all(np.isfinite(result.records[0].clean_measurements))
    assert result.records[0].clean_measurements.sum() == pytest.approx(0.0, abs=1e-12)
