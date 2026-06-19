# Benchmark

## Purpose

`benchmark` — инфраструктура forward-only experiments:

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

Модуль не строит Green functions и не решает inverse problem. Он генерирует воспроизводимые synthetic observations, которые смогут использовать будущие слои.

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

## Relation to future inverse benchmark

Сейчас records являются только synthetic forward observations. Будущие Green/inverse modules смогут читать их как benchmark inputs, но соответствующие алгоритмы намеренно не входят в `benchmark`.
