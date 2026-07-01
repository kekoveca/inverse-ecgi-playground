# fem

Полная документация: [../docs/fem.md](../docs/fem.md).

`fem` - это FEniCSx-backed слой для FEM-задачи Пуассона с чистыми условиями Неймана. Он отвечает только за универсальное FEM-ядро: function space, stiffness matrix, nullspace, проверку правых частей и решение линейных систем.

Модуль не знает деталей обратной задачи, дипольной модели, электродных измерений или Green-функций. Эти сущности должны строиться выше по стеку и передавать в `fem` только геометрию и правые части.

## Ответственность

- создать DOLFINx mesh из `geometry.MeshData`;
- создать scalar function space;
- собрать stiffness matrix `K`;
- настроить constant nullspace для чистой задачи Неймана;
- проверять совместимость RHS с nullspace;
- решать `K u = b` для разных RHS без повторной сборки `K`;
- возвращать `dolfinx.fem.Function`;
- приводить найденный потенциал к фиксированной калибровке.

## Основные объекты

### `FunctionSpaceFactory`

Создает DOLFINx mesh и function space.

В MVP поддерживается только:

```text
P1 Lagrange
```

Попытка создать задачу с `degree != 1` завершится `ValueError`.

### `StiffnessOperator`

Собирает и хранит основной FEM-артефакт:

```text
K_ij = integral sigma grad(phi_i) . grad(phi_j) dx
```

Матрица `K` собирается один раз для одной геометрии и затем переиспользуется.

Текущие ограничения MVP:

- `sigma` - scalar;
- `sigma` - constant;
- boundary conditions в сборке матрицы не задаются, потому что рассматривается чистая задача Неймана.

### `NeumannNullspaceHandler`

Инкапсулирует constant nullspace PETSc для чистой задачи Неймана.

Он умеет:

- создать PETSc `NullSpace(constant=True)`;
- прикрепить nullspace к матрице `K`;
- проверить nullspace на матрице;
- удалить constant-компоненту из RHS;
- проверить совместимость RHS;
- зафиксировать gauge решения вычитанием среднего значения.

`ConstantNullspace` оставлен как обратносовместимое имя для `NeumannNullspaceHandler`.

### `LinearSolver`

Обертка над PETSc KSP.

Настраивает solver один раз на уже собранной матрице `K` и затем решает системы с разными правыми частями.

Поддерживаемые типы задаются литералами:

```python
KSPType = Literal["cg", "gmres", "preonly"]
PCType = Literal["hypre", "gamg", "jacobi", "lu", "none"]
```

### `FEMProblem`

Главный объект модуля. Он собирает все роли вместе:

1. принимает `MeshData` или `TorsoGeometry`;
2. берет volume mesh;
3. создает function space;
4. собирает `K`;
5. прикрепляет Neumann nullspace;
6. настраивает `LinearSolver`;
7. решает много RHS через один и тот же `K`.

`NeumannPoissonSolver` - обратносовместимое имя для `FEMProblem`.

### `SolverDiagnostics`

Хранит диагностическую информацию последнего solve:

```python
SolverDiagnostics(
    ksp_type="cg",
    pc_type="hypre",
    converged_reason=...,
    residual_norm=...,
    nullspace_test_passed=...,
)
```

### `DOLFINxP1Mapping`

Coordinate-verified serial mapping между MeshData node ids и scalar P1 DOLFINx dofs. `FEMProblem.p1_node_dof_mapping()` кэширует обе перестановки; `node_to_dof_map` и `dof_to_node_map` доступны как properties.

### `DOLFINxP1TetraLocator`

Кэшированный locator owned local tetra cells. Он хранит dof coordinates, cell dofs/vertices/centers и KD-tree, а попадание подтверждает barycentric test:

```python
locator = problem.p1_tetra_locator()
cell_ids = locator.locate_points(points)
grads_phi = locator.basis_gradients(cell_ids)
```

Locator не смешивает MeshData cell ids с local DOLFINx cell ids и поддерживает только scalar P1 tetra spaces.

## Основной артефакт: `K`

`K` - FEM stiffness matrix. В `FEMProblem` она доступна как:

```python
problem.K
```

Для совместимости со старым кодом также есть:

```python
problem.A
```

`K` должна переиспользоваться для:

- прямых задач от разных диполей;
- построения Green-функций;
- тестов согласованности;
- любых серий RHS на одной и той же геометрии.

## Минимальный пример

```python
import numpy as np

from fem import FEMProblem
from geometry import MeshData

points = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
cells = np.array([[0, 1, 2, 3]], dtype=np.int64)

mesh = MeshData(points=points, cells=cells, cell_type="tetra")

problem = FEMProblem(
    mesh,
    sigma=1.0,
    ksp_type="cg",
    pc_type="hypre",
)

rhs = problem.rhs_from_local_array(np.zeros(problem.zero_function().x.array.shape))
u = problem.solve(rhs)

print(problem.K)
print(problem.diagnostics)
```

В реальном RHS массив должен быть совместим с распределением dof в DOLFINx/PETSc. `rhs_from_local_array()` - это удобный helper для тестов и прототипов.

## RHS и чистая задача Неймана

Для чистой задачи Неймана RHS должен быть совместим с constant nullspace:

```text
sum(b) = 0
```

По умолчанию `solve()` сначала удаляет constant-компоненту RHS:

```python
u = problem.solve(rhs, remove_nullspace_component=True)
```

Затем RHS проверяется:

```python
problem.check_rhs_compatible(rhs)
```

Если совместимость можно проверить и residual больше допуска, будет выброшен `ValueError`.

## Gauge fixing

Решение чистой задачи Неймана определено с точностью до константы. Поэтому после решения `solve()` по умолчанию фиксирует gauge:

```python
u = problem.solve(rhs, fix_gauge=True)
```

Текущая калибровка - вычитание среднего значения потенциала из локального массива функции. Если доступен PETSc Vec с `sum()` и `getSize()`, используется среднее по PETSc-вектору.

## Многократные решения

`FEMProblem` предназначен для многократного решения на одной геометрии:

```python
problem = FEMProblem(mesh)

for rhs in rhs_list:
    u = problem.solve(rhs)
    print(problem.diagnostics.residual_norm)
```

Матрица `K` и KSP setup при этом не пересобираются.

## Освобождение ресурсов

PETSc-объекты нужно явно уничтожить, когда задача больше не нужна:

```python
problem.destroy()
```

Это уничтожает KSP и matrix owner (`StiffnessOperator`).

Также сбрасываются cached node↔DOF mapping и P1 tetra locator. Objects из этих кэшей нельзя использовать после `destroy()`.

## Зависимости

FEniCSx импортируется лениво через `require_fenicsx()`.

Для реального solve нужны:

- `dolfinx`;
- `basix`;
- `ufl`;
- `mpi4py`;
- `petsc4py`.

Если стек не установлен, модуль выбрасывает понятный `ImportError` с сообщением о необходимости FEniCSx-окружения.

## Тесты с настоящим DOLFINx

Обычные unit-тесты используют fake-объекты и не требуют FEniCSx:

```bash
pytest test_fem_units.py
```

Реальные integration-тесты лежат в `test_fem_dolfinx_integration.py`. Они создают настоящий DOLFINx mesh, собирают матрицу `K`, проверяют nullspace и решают две системы с разными RHS. По умолчанию они пропускаются, чтобы обычный `pytest` не падал в окружениях без рабочего MPI.

Запуск в WSL:

```bash
TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp RUN_DOLFINX_TESTS=1 pytest test_fem_dolfinx_integration.py
```

Полный прогон вместе с integration-тестами:

```bash
TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp RUN_DOLFINX_TESTS=1 pytest
```

## Что не входит в `fem`

`fem` не должен знать:

- как строится дипольный RHS;
- как устроены электроды;
- какие Green-функции нужны обратной задаче;
- какие измерения сравниваются;
- как регуляризуется inverse problem.

Эти задачи должны жить выше, а `fem` должен оставаться переиспользуемым backend-слоем для линейных FEM-систем.
