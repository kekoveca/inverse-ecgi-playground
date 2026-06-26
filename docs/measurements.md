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
- `central_project_electrodes_to_surface(...)` центрально проецирует внешние электроды на surface mesh или на boundary, извлеченный из tetra volume mesh.

Геометрия тетраэдра переиспользуется из `sources`; отдельной копии barycentric math в `measurements` нет.

Возвращаемые cell ids принадлежат MeshData ordering.

`locate_points_in_tetra_mesh` остается строгим и выбрасывает `ValueError` для точек вне volume mesh. Для электродов доменная логика мягче: `build_measurement_operator` по умолчанию вызывает центральную проекцию для электродов, которые не лежат внутри торса. Проекция строит луч от центра volume mesh через внешний электрод и берет первое пересечение с surface mesh. Если `surface_mesh` не передан, boundary triangles извлекаются из tetra mesh.

```python
from measurements import central_project_electrodes_to_surface

projected_electrodes, report = central_project_electrodes_to_surface(
    volume_mesh=volume_mesh,
    electrodes=electrodes,
    surface_mesh=surface_mesh,  # optional
)

print(report.projected_indices)
print(report.max_projection_distance)
```

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
    surface_mesh=surface_mesh,  # optional
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

Если электроды были спроецированы, summary доступен в `op.metadata["electrode_projection"]`.

## Constant potential test

Average reference должен уничтожать константу:

```python
u_constant = np.full(volume_mesh.num_points, 5.0)
g = op.evaluate(u_constant)
assert np.allclose(g, 0.0)
```

Это соответствует инвариантности электродных измерений к gauge чистой задачи Неймана.
