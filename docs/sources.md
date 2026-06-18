# Sources

`sources` реализует точечный дипольный источник для scalar P1 FEM на тетраэдральной сетке. Геометрические вычисления доступны в numpy; отдельный адаптер собирает RHS в DOLFINx ordering.

## PointDipole

```python
from sources import PointDipole

source = PointDipole(
    position=[0.0, 0.0, 0.0],
    moment=[0.0, 0.0, 1.0],
    cell_id=None,
    name="point_dipole",
    metadata={},
)
```

Поля:

- `position` — координата источника, shape `(3,)`;
- `moment` — дипольный момент, shape `(3,)`;
- `cell_id` — optional MeshData cell id для numpy workflows;
- `name`, `metadata` — пользовательские метаданные.

`with_cell_id` возвращает новый immutable объект. `normalized_moment` возвращает направление ненулевого момента.

## P1 tetra geometry

Модуль предоставляет:

- `tetra_signed_volume`, `tetra_volume`;
- `barycentric_coordinates_tetra`;
- `point_in_tetra`;
- `gradients_p1_tetra`.

`gradients_p1_tetra(vertices)` возвращает `grads[a] = grad(phi_a)` в порядке переданных четырёх вершин.

## Numpy RHS

```python
from sources import assemble_point_dipole_rhs_numpy

rhs_numpy = assemble_point_dipole_rhs_numpy(volume_mesh, source)
```

Эта функция собирает

```text
b_i = moment . grad(phi_i)
```

в **MeshData node ordering**. Она предназначена для numpy-тестов, геометрической проверки и workflows без FEniCSx.

Если `source.cell_id` задан, numpy assembler интерпретирует его как MeshData cell id.

## PETSc/FEniCSx RHS

```python
from sources import assemble_point_dipole_rhs_petsc

rhs = assemble_point_dipole_rhs_petsc(solver, source)
potential = solver.solve(rhs)
```

PETSc adapter не копирует numpy RHS. Он:

1. находит `source.position` среди локальных DOLFINx cells;
2. получает `cell_dofs = solver.V.dofmap.cell_dofs(cell_id)`;
3. берёт dof coordinates в том же локальном порядке;
4. вычисляет четыре local values;
5. записывает их непосредственно в DOLFINx Function.

По умолчанию `source.cell_id` **не считается DOLFINx cell id**. Это поле относится к MeshData ordering.

Явный параметр `cell_id=` в `assemble_point_dipole_rhs_petsc` означает DOLFINx cell id. Флаг `trust_source_cell_id=True` разрешает использовать `source.cell_id` как DOLFINx id, но только после проверки ordering.

## Sign convention

Текущая конвенция:

```python
local_rhs = gradients_p1_tetra(vertices) @ source.moment
```

Знак намеренно не меняется в диагностических исправлениях. Финальная физическая проверка знака будет выполнена позже через FEM/Green consistency.

## Source location debugging

### DOLFINx point location

```python
from sources import locate_point_in_dolfinx_p1_tetra_mesh

cell_id = locate_point_in_dolfinx_p1_tetra_mesh(solver, source.position)
```

MVP-функция перебирает owned local cells и возвращает local DOLFINx cell id.

### Full diagnostics

```python
from sources import inspect_point_dipole_location_petsc

info = inspect_point_dipole_location_petsc(solver, source)
print(info["used_cell_id"])
print(info["cell_dofs"])
print(info["barycentric_in_dolfinx_cell"])
print(info["is_inside_used_dolfinx_cell"])
print(info["ordering_warning"])
```

`inspect_point_dipole_rhs_petsc` является обратносовместимым именем той же полной диагностики.

Проверка RHS:

```python
info = inspect_point_dipole_rhs_petsc(solver, source)
print(info["nonzero_dofs"])
print(info["local_rhs"])
print(info["local_rhs_sum"])
```

Для общего момента на P1 tetra ожидаются четыре cell dofs; отдельные значения могут математически оказаться нулевыми для специальных направлений момента.

### Compare cell ids geometrically

```python
from sources import compare_meshdata_and_dolfinx_cell_centers

report = compare_meshdata_and_dolfinx_cell_centers(solver, max_cells=1000)
print(report["max_diff"])
print(report["worst_cell_id"])
```

Большие differences показывают, что одинаковые integer ids называют разные cells.

### Source marker

```python
from forward import export_dolfinx_function_to_vtx
from sources import create_cell_marker_function

marker = create_cell_marker_function(solver, info["used_cell_id"])
export_dolfinx_function_to_vtx(
    marker,
    "output/source_marker.bp",
    name="source_marker",
)
```

Откройте `source_marker.bp` в ParaView для визуальной проверки фактически использованной ячейки.
