# Measurements

## Purpose

`measurements` преобразует nodal P1 potential в значения на электродах и применяет reference-систему.

```text
nodal values u -> raw electrode values y_raw -> referenced values g
```

Основная реализация использует numpy/scipy и не требует DOLFINx.

## Point location

- `locate_points_in_tetra_mesh(mesh, points)` возвращает MeshData cell ids и barycentric coordinates.
- `locate_electrodes_in_mesh(mesh, electrodes)` применяет тот же алгоритм к `ElectrodeSet.positions`.

Геометрия тетраэдра переиспользуется из `sources`; отдельной копии barycentric math в `measurements` нет.

Возвращаемые cell ids принадлежат MeshData ordering.

## Interpolation matrix

Для P1 tetra:

```text
y_raw = P u
```

Каждая строка `P` содержит четыре barycentric weights в columns узлов содержащего тетраэдра.

```python
from measurements import build_point_interpolation_matrix

P = build_point_interpolation_matrix(
    mesh=volume_mesh,
    points=electrodes.positions,
    sparse=True,
)
```

При `sparse=True` возвращается scipy CSR matrix. Если scipy недоступен, функция предупреждает и возвращает dense array.

`u` для этой матрицы должен быть представлен в **MeshData node ordering**. DOLFINx DOF ordering нельзя считать совпадающим без проверенного mapping.

## Reference systems

Поддерживаются:

- `none`: `g = y`;
- `average`: `g = y - mean(y)`;
- `single`: `g_i = y_i - y_reference_index`.

```python
from measurements import apply_reference

g_average = apply_reference(y_raw, reference="average")
g_single = apply_reference(y_raw, reference="single", reference_index=0)
```

Reference matrix обозначается `R`.

## MeasurementOperator

Полный линейный оператор:

```text
M = R @ P
g = M u
```

```python
from measurements import build_measurement_operator

op = build_measurement_operator(
    mesh=volume_mesh,
    electrodes=electrodes,
    reference="average",
    sparse=True,
)

y_raw = op.evaluate_raw(nodal_values)
g = op.evaluate(nodal_values)
P = op.raw_matrix()
M = op.matrix()
```

Методы:

- `raw_matrix()` — `P`;
- `matrix()` — `M = R @ P`;
- `evaluate_raw()` — raw electrode values;
- `evaluate()` — referenced values.

Строки `M` используются как RHS vectors в модуле `green`: `K G_i = M_i^T`. Перед записью в DOLFINx Function они отображаются из MeshData node ordering в DOLFINx DOF ordering.

## Constant potential test

Average reference должен уничтожать константу:

```python
u_constant = np.full(volume_mesh.num_points, 5.0)
g = op.evaluate(u_constant)
assert np.allclose(g, 0.0)
```

Это соответствует инвариантности электродных измерений к gauge чистой задачи Неймана.
