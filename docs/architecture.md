# Architecture

## Pipeline

```text
geometry
  |
  v
fem
  |
  v
sources
  |
  v
measurements
  |
  v
forward
  |
  v
ParaView / diagnostics
```

Reciprocal layer:

```text
measurements M
  |
  v
green: K G_i = M_i^T
  |
  v
dipole transfer matrix A
  |
  v
inverse (future)
```

Experimental layer:

```text
geometry / fem / sources / measurements / forward
                         |
                         v
                     benchmark
```

Это схема потока данных, а не жёсткая цепочка импортов. Например, `sources` содержит numpy-геометрию и адаптер к уже созданному FEM solver, а `measurements` может работать полностью без DOLFINx.

## Module responsibilities

### geometry

Хранит геометрию без зависимости от FEniCSx: узлы, connectivity, physical tags, электроды, source region и агрегат `TorsoGeometry`.

### fem

Преобразует `MeshData` в DOLFINx mesh, создаёт scalar P1 function space, собирает stiffness matrix `K`, настраивает PETSc constant nullspace и решает системы с несколькими RHS.

### sources

Описывает `PointDipole`, геометрию P1 tetra и сборку дипольного RHS. Numpy-сборка использует MeshData node ordering; FEM-адаптер собирает значения непосредственно в DOLFINx DOF ordering.

### measurements

Строит interpolation matrix `P` и reference matrix `R`:

```text
y_raw = P u
g = R P u
```

### forward

Компонует готовые слои:

```text
source -> rhs -> solve -> nodal values -> measurements -> ForwardResult -> export
```

### green

Преобразует строки `M = R @ P` в совместимые Neumann RHS, решает Green-задачи на той же stiffness matrix и собирает `A[j, i, :] = grad G_i(x_j)`. Transfer matrix предсказывает измерения как `g = A_j p`; знак контролируется FEM/Green consistency diagnostic.

### benchmark

Комбинирует geometry, source sets, electrode subsets и noise models в forward-only experiments. Сохраняет clean/noisy measurements и scalar metrics; интеграция с Green и inverse остаётся отдельным следующим этапом.

## Important separation

- `geometry` не импортирует FEniCSx и остаётся пригодным для preprocessing и numpy-тестов.
- `fem` владеет DOLFINx/PETSc solver objects и временем жизни матрицы/KSP.
- `sources` и `measurements` имеют numpy-only core. Их FEM-адаптеры принимают уже созданные DOLFINx-объекты.
- `forward` не пересобирает матрицу: он использует существующий `NeumannPoissonSolver`. Экспорт через `dolfinx.io` импортируется лениво.
- `green` использует numpy/scipy measurement matrices, но создаёт Green RHS через проверенный node-to-dof mapping.

## Ordering boundaries

### Nodes and DOFs

`MeshData.points[node_id]` и `dolfinx.fem.Function.x.array[dof_id]` используют разные пространства индексов. Совпадение integer ids не гарантируется.

Нельзя копировать numpy-вектор, собранный по `MeshData node_id`, напрямую в PETSc/FEniCSx vector без проверенного отображения.

### Cells

`MeshData cell_id` и локальный `DOLFINx cell_id` также могут различаться. На реальной `torso.msh` это различие наблюдается после создания DOLFINx mesh.

Поэтому PETSc RHS точечного диполя по умолчанию заново ищет ячейку по `source.position` в DOLFINx ordering. Явный аргумент `cell_id` для PETSc-адаптера означает DOLFINx cell id.

## Physical tags

Внутренняя конвенция:

```python
field_data: dict[str, tuple[int, int]]
# name -> (dim, tag)
```

Она соответствует порядку Gmsh API. `meshio` возвращает `(tag, dim)`, и `read_gmsh_meshio` преобразует пары при импорте.

## Pure Neumann problem

Stiffness matrix чистой задачи Неймана имеет константное ядро. Потенциал определён с точностью до добавления константы. `fem` прикрепляет PETSc `NullSpace`, проверяет/проецирует RHS и после решения фиксирует gauge вычитанием среднего.

Average-reference измерений дополнительно инвариантен к константному сдвигу потенциала.

## Verification layer

`verification` не участвует в production pipeline. Он предоставляет unit-cube meshes, analytic/manufactured functions и convergence reports для тестов FEM/forward. Это разделение позволяет проверять smooth FEM convergence независимо от сингулярного point-dipole solution.
