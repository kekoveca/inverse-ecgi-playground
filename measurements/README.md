# measurements

Полная документация: [../docs/measurements.md](../docs/measurements.md).

`measurements` преобразует FEM-потенциал в узлах P1 тетраэдральной сетки в вектор значений на электродах.

Модуль не зависит от DOLFINx в основной реализации: он работает с `geometry.MeshData`, `geometry.ElectrodeSet`, `numpy` и, если доступен, `scipy.sparse`.

## Матрицы

Для электродов строится interpolation matrix `P`:

```text
y_raw = P u
```

Для P1 tetra каждая строка `P` содержит четыре ненулевых значения: барицентрические координаты электрода в найденной ячейке.

Reference-система задается матрицей `R`. Для average reference:

```text
R = I - 1/N 11^T
g = R y_raw
```

Полный measurement operator:

```text
M = R @ P
g = M u
```

## Минимальный пример

```python
import numpy as np

from geometry import ElectrodeSet, MeshData
from measurements import build_measurement_operator

volume_mesh = MeshData(
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
    positions=np.array(
        [
            [1.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ]
    ),
    labels=["E1", "E2"],
)

op = build_measurement_operator(
    mesh=volume_mesh,
    electrodes=electrodes,
    reference="average",
)

u = np.array([0.0, 1.0, 2.0, 3.0])
y_raw = op.evaluate_raw(u)
g = op.evaluate(u)
M = op.matrix()
```

Поддержанные reference-системы:

- `"none"`: без изменения значений;
- `"average"`: вычитание среднего по электродам;
- `"single"`: вычитание значения выбранного электрода `reference_index`.
