# Architecture

## Pipeline

```text
geometry
  |
  v
fem
  |
  v
sources + measurements
  |
  v
forward
  |
  v
green
  |
  v
inverse
  |
  v
benchmark
  |
  v
performance / examples / ParaView diagnostics
```

The arrows describe data flow rather than a strict import chain. The forward branch solves `K u = b_source` and evaluates `g = R P u`; the reciprocal branch solves `K G_i = M_i^T` and builds the transfer tensor used by inverse.

Benchmark layer:

```text
geometry / fem / sources / measurements / forward
                         |
                         v
                     benchmark
```

Supporting layers:

```text
verification -> convergence and consistency tests
performance  -> timing, memory and scaling profiles
```

Это схема потока данных, а не жёсткая цепочка импортов. Например, `sources` содержит numpy-геометрию и адаптер к уже созданному FEM solver, а `measurements` может работать полностью без DOLFINx.

## Module responsibilities

### geometry

Хранит геометрию без зависимости от FEniCSx: узлы, connectivity, physical tags, электроды, source region и агрегат `TorsoGeometry`.

### fem

Преобразует `MeshData` в DOLFINx mesh, создаёт scalar P1 function space, собирает stiffness matrix `K`, настраивает PETSc constant nullspace и решает системы с несколькими RHS. Здесь же находятся кэшируемые node↔DOF mapping и local-cell `DOLFINxP1TetraLocator`.

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

### inverse

Использует готовый `GreenTransferMatrix` и observed measurements. Для каждого candidate решает маленькую Tikhonov/LS-задачу на три компоненты момента и выбирает минимальный residual. Модуль не реализует multiple dipoles и не меняет sign convention.

### benchmark

Комбинирует geometry, source sets, electrode subsets и noise models в forward experiments. Inverse benchmark consumes `ForwardBenchmarkResult` плюс `GreenTransferMatrix` и сохраняет localization/moment/residual metrics.

### verification

Содержит smooth manufactured solution, unit-cube refinement и convergence-report utilities. Модуль не участвует в production solve.

### performance

Предоставляет timers, memory snapshots, report writers и profiling CLIs. Он измеряет существующие stages и не меняет математическое поведение.

## Important separation

- `geometry` не импортирует FEniCSx и остаётся пригодным для preprocessing и numpy-тестов.
- `fem` владеет DOLFINx/PETSc solver objects и временем жизни матрицы/KSP.
- `sources` и `measurements` имеют numpy-only core. Их FEM-адаптеры принимают уже созданные DOLFINx-объекты.
- `forward` не пересобирает матрицу: он использует существующий `NeumannPoissonSolver`. Экспорт через `dolfinx.io` импортируется лениво.
- `green` использует numpy/scipy measurement matrices, но создаёт Green RHS через проверенный node-to-dof mapping.
- `inverse` не зависит от DOLFINx напрямую: он работает с numpy transfer matrices и measurement vectors.
- `benchmark` не строит GreenTransferMatrix внутри inverse runner; он принимает готовый transfer, чтобы sweeps по noise/electrodes/lambda не пересобирали Green-базис без необходимости.

## Data ownership and ordering boundaries

### Nodes and DOFs

`MeshData.points[node_id]` и `dolfinx.fem.Function.x.array[dof_id]` используют разные пространства индексов. Совпадение integer ids не гарантируется.

Нельзя копировать numpy-вектор, собранный по `MeshData node_id`, напрямую в PETSc/FEniCSx vector без проверенного отображения.

### Cells

`MeshData cell_id` и локальный `DOLFINx cell_id` также могут различаться. На реальной `torso.msh` это различие наблюдается после создания DOLFINx mesh.

Поэтому PETSc RHS точечного диполя по умолчанию ищет ячейку по `source.position` в DOLFINx ordering. Поиск использует кэшированный `DOLFINxP1TetraLocator`: KD-tree по центрам задаёт кандидатов, а барицентрическая проверка подтверждает принадлежность. Явный аргумент `cell_id` для PETSc-адаптера означает local DOLFINx cell id.

`SourceRegion.candidate_cell_ids` всегда хранит MeshData cell ids. В `GreenTransferMatrix.candidate_cell_ids` хранятся уже найденные local DOLFINx cell ids. Эти массивы нельзя взаимозаменять.

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

## Units and coordinates

`MeshData.points`, `ElectrodeSet.positions`, `SourceRegion.candidate_points` и `PointDipole.position` должны использовать одну coordinate frame и одни единицы. Значение `sigma`, localization thresholds и distance diagnostics интерпретируются в согласованных с этой системой единицах; автоматической конверсии mm/m нет.

## Cached spatial data

Один `NeumannPoissonSolver` владеет кэшами, связанными с созданным DOLFINx mesh:

- `p1_node_dof_mapping()` — serial scalar-P1 permutation MeshData node ↔ DOLFINx dof;
- `p1_tetra_locator()` — dof coordinates, local cell dofs/vertices/centers и KD-tree;
- stiffness matrix и KSP setup для всех forward/Green RHS.

Кэши действительны только для этого solver и очищаются в `solver.destroy()`. Текущий locator ищет owned local cells; distributed global point ownership остаётся отдельной задачей.

## Verification layer

`verification` не участвует в production pipeline. Он предоставляет unit-cube meshes, analytic/manufactured functions и convergence reports для тестов FEM/forward. Это разделение позволяет проверять smooth FEM convergence независимо от сингулярного point-dipole solution.
