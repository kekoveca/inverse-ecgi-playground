# Benchmark

## Purpose

`benchmark` — инфраструктура forward and inverse experiments:

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

Модуль не строит Green functions сам. Forward benchmark генерирует воспроизводимые synthetic observations, а inverse benchmark применяет готовый `GreenTransferMatrix` к этим observations и считает localization/moment metrics.

## Scenario

`ForwardBenchmarkScenario` описывает декартово произведение sources, electrode subsets и noise models для одной geometry:

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

`to_config_dict()` сохраняет counts/configuration, но не сериализует mesh arrays или measurements.

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

Также доступна deterministic генерация `generate_sources_from_region` с `stride` и `max_positions`. `SourceSet.to_table()` возвращает pandas/CSV-friendly rows.

`source.cell_id` остаётся MeshData cell id. PETSc assembler по умолчанию заново локализует `source.position` в DOLFINx ordering.

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

Farthest-point sampling жадно максимизирует минимальное расстояние до уже выбранного множества. `ElectrodeSubset.indices` сохраняет ids исходного `ElectrodeSet`.

## Noise

```python
from benchmark import NoNoise, RelativeGaussianNoise

noise_models = [
    NoNoise(),
    RelativeGaussianNoise(snr_db=40.0, seed=0),
    RelativeGaussianNoise(snr_db=30.0, seed=1),
]
```

Доступны:

- `NoNoise`;
- `AbsoluteGaussianNoise(sigma, seed)`;
- `RelativeGaussianNoise(snr_db, seed)`.

Relative noise масштабируется по L2 norm до заданного amplitude SNR. Для нулевого signal возвращается нулевой noise. `apply` никогда не изменяет input inplace.

## Metrics

Основные функции:

- `rmse`;
- `relative_l2_error`;
- `max_abs_error`;
- `correlation`;
- `compute_snr_db`;
- `forward_signal_metrics`;
- `noise_metrics`.

Correlation почти константного вектора возвращает `NaN`; zero-noise SNR равен infinity.

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

Runner создаёт один Poisson solver на geometry и переиспользует stiffness matrix для всех subsets/sources. По умолчанию potentials не сохраняются. `export_potentials=True` экспортирует VTX/BP, но требует `output_dir`.

## Outputs

```text
results/torso_forward_smoke/
├── config.json
├── records.csv
├── measurements.npz
└── summary.json
```

- `config.json` — compact scenario configuration;
- `records.csv` — scalar metadata и metrics без массивов;
- `measurements.npz` — `clean_XXXXXX`, `noisy_XXXXXX`, `noise_XXXXXX` для каждого record;
- `summary.json` — counts и использованные subsets/models.

Отдельные NPZ keys поддерживают разные количества электродов без unsafe object arrays.

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
