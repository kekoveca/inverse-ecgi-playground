# Benchmark

## Purpose

`benchmark` provides infrastructure for forward and inverse experiments:

```text
geometry
  x source set
  x electrode subset
  x noise model
  -> forward solve
  -> clean/noisy measurements
  -> metrics
  -> saved result
```

The module does not build Green functions itself. The forward benchmark generates reproducible synthetic observations, while the inverse benchmark applies an existing `GreenTransferMatrix` and computes localization/moment metrics.

## Scenario

`ForwardBenchmarkScenario` describes the Cartesian product of sources, electrode subsets, and noise models for one geometry:

```python
from benchmark import ForwardBenchmarkScenario

scenario = ForwardBenchmarkScenario(
    name="torso_forward_smoke",
    geometry=geometry,
    sources=sources.sources,
    electrode_sets=subsets,
    noise_models=noise_models,
    reference="average",
    metadata={"purpose": "forward_only"},
)
scenario.validate()
```

`to_config_dict()` stores counts/configuration without serializing mesh arrays or measurements.

## Source sets

```python
from benchmark import axis_moments, generate_random_sources_from_region

sources = generate_random_sources_from_region(
    geometry.source_region,
    n_positions=10,
    moments=axis_moments(),
    seed=42,
)
```

Deterministic `generate_sources_from_region` is also available with `stride` and `max_positions`. `SourceSet.to_table()` returns pandas/CSV-friendly rows.

`source.cell_id` remains a MeshData cell id. By default, the PETSc assembler locates `source.position` again in DOLFINx ordering.

## Electrode subsets

```python
from benchmark import (
    make_all_electrodes_subset,
    select_farthest_point_electrodes,
    select_random_electrodes,
)

subsets = [
    make_all_electrodes_subset(geometry.electrodes),
    select_random_electrodes(geometry.electrodes, n=32, seed=1),
    select_farthest_point_electrodes(geometry.electrodes, n=32, seed=1),
]
```

Farthest-point sampling greedily maximizes the minimum distance to the selected set. `ElectrodeSubset.indices` preserves ids from the original `ElectrodeSet`.

## Noise

```python
from benchmark import NoNoise, RelativeGaussianNoise

noise_models = [
    NoNoise(),
    RelativeGaussianNoise(snr_db=40.0, seed=0),
    RelativeGaussianNoise(snr_db=30.0, seed=1),
]
```

Available models:

- `NoNoise`;
- `AbsoluteGaussianNoise(sigma, seed)`;
- `RelativeGaussianNoise(snr_db, seed)`.

Relative noise is scaled by L2 norm to the requested amplitude SNR. A zero signal produces zero noise. `apply` never mutates its input in place.

## Metrics

Main functions:

- `rmse`;
- `relative_l2_error`;
- `max_abs_error`;
- `correlation`;
- `compute_snr_db`;
- `forward_signal_metrics`;
- `noise_metrics`.

Correlation of an almost constant vector returns `NaN`; zero-noise SNR is infinity.

## Running

```python
from benchmark import ForwardBenchmarkRunner, save_forward_benchmark_result
from fem import NeumannPoissonSolver

runner = ForwardBenchmarkRunner(
    poisson_solver_factory=lambda mesh: NeumannPoissonSolver(
        mesh,
        degree=1,
        sigma=1.0,
    ),
)

result = runner.run(scenario)
save_forward_benchmark_result(result, "results/torso_forward_smoke")
```

The runner creates one Poisson solver per geometry and reuses the stiffness matrix for all subsets/sources. Potentials are not retained by default. `export_potentials=True` exports VTX/BP and requires `output_dir`.

## Outputs

```text
results/torso_forward_smoke/
├── config.json
├── records.csv
├── measurements.npz
└── summary.json
```

- `config.json` — compact scenario configuration;
- `records.csv` - scalar metadata and metrics without arrays;
- `measurements.npz` - `clean_XXXXXX`, `noisy_XXXXXX`, and `noise_XXXXXX` for each record;
- `summary.json` - counts and selected subsets/models.

Separate NPZ keys support different electrode counts without unsafe object arrays.

## Inverse benchmark

Inverse benchmark consumes a `ForwardBenchmarkResult` and a matching `GreenTransferMatrix`:

```text
ForwardBenchmarkResult
  + GreenTransferMatrix
  -> InverseBenchmarkRunner
  -> InverseBenchmarkResult
```

```python
from benchmark import (
    InverseBenchmarkScenario,
    InverseBenchmarkRunner,
    filter_forward_result_by_electrode_set,
    save_inverse_benchmark_result,
)

filtered = filter_forward_result_by_electrode_set(
    forward_result,
    electrode_set_name="all",
)

scenario = InverseBenchmarkScenario(
    name="single_dipole_inverse",
    forward_result=filtered,
    transfer_matrix=transfer,
    lambda_reg=1e-10,
    localization_threshold=20.0,
    use_clean_measurements=True,
    use_noisy_measurements=True,
    reference="average",
)

inverse_result = InverseBenchmarkRunner().run(scenario)
save_inverse_benchmark_result(
    inverse_result,
    "results/single_dipole_inverse",
)
```

Convenience wrapper:

```python
from benchmark import run_inverse_benchmark

inverse_result = run_inverse_benchmark(
    filtered,
    transfer,
    lambda_reg=1e-10,
    localization_threshold=20.0,
)
```

Outputs:

```text
results/single_dipole_inverse/
├── inverse_config.json
├── inverse_records.csv
└── inverse_summary.json
```

`inverse_records.csv` contains true source data, estimated source data, residuals, localization error, moment error and optional success flag.

Important restriction: one `InverseBenchmarkScenario` corresponds to one electrode subset and one `GreenTransferMatrix`. If a forward benchmark contains several electrode subsets, filter records before running inverse benchmark.

Large residual maps and candidate moment maps are not saved by default. Store transfer matrices via the `green` cache when they should be reused across sweeps.

## Reproducibility and transfer provenance

Noise models own deterministic seeds, and scenario configs avoid serializing large mesh/measurement arrays. For cached `GreenTransferMatrix` files, include at least the following user metadata:

- geometry/mesh identifier or hash;
- electrode subset labels/order;
- reference and measurement row indices;
- conductivity `sigma`;
- candidate source-region identifier and coordinate units;
- transfer sign and code/version identifier when available.

The current cache round-trips metadata but does not enforce this provenance schema. `InverseBenchmarkScenario` validates measurement length and one-electrode-subset usage; it cannot prove that a numerically compatible transfer came from the same geometry.
