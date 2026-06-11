# sources

`sources` содержит независимую от DOLFINx numpy-реализацию точечного дипольного источника для P1 FEM на тетраэдральной сетке.

Для диполя с положением `x0` и моментом `p`, лежащего в тетраэдре `T`, локальная правая часть задается как:

```text
b_i = p . grad(phi_i)|_T
```

Для P1 тетраэдра ненулевые значения есть только на четырех вершинах ячейки. Так как сумма градиентов P1 базисных функций равна нулю, сумма RHS также равна нулю, что дает совместимость с чистой задачей Неймана.

## Минимальный пример

```python
import numpy as np

from geometry import MeshData
from sources import PointDipole, assemble_point_dipole_rhs_numpy

mesh = MeshData(
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

source = PointDipole(
    position=[0.25, 0.25, 0.25],
    moment=[1.0, 0.0, 0.0],
)

rhs = assemble_point_dipole_rhs_numpy(mesh, source)
```

Если `source.cell_id` не задан, ячейка источника ищется через `locate_points_in_mesh(...)`: сначала кандидаты упорядочиваются `scipy.spatial.cKDTree` по центроидам тетраэдров, затем принадлежность подтверждается барицентрической проверкой.

Текущий знак RHS выбран как:

```python
local_rhs = gradients_p1_tetra(vertices) @ source.moment
```

Этот знак позже должен быть окончательно проверен на этапе FEM-Green consistency.

## FEM adapter

Для текущего `fem.FEMProblem` / `fem.NeumannPoissonSolver` есть адаптер:

```python
from sources import assemble_point_dipole_rhs_petsc

rhs_function = assemble_point_dipole_rhs_petsc(problem, source)
u = problem.solve(rhs_function)
```

Адаптер использует `problem.mesh_data` и `problem.rhs_from_local_array(...)`.
