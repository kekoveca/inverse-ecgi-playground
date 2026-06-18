# sources

Полная документация: [../docs/sources.md](../docs/sources.md).

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

Адаптер использует `problem.mesh_data`, `problem.V` и `problem.zero_function()`. Он находит ячейку по `source.position` в локальном DOLFINx cell ordering и записывает локальный вклад в `V.dofmap.cell_dofs(cell_id)`. Numpy-вектор в порядке узлов `MeshData` не копируется.

## Debugging point dipole RHS

`assemble_point_dipole_rhs_numpy(...)` собирает RHS в порядке узлов `MeshData`. Это удобно для numpy-тестов и геометрической проверки, но этот порядок нельзя считать равным порядку dof в FEniCSx.

`assemble_point_dipole_rhs_petsc(...)` собирает RHS в FEniCSx DOF ordering. Для P1 tetra он:

1. находит ячейку источника;
2. берет `cell_dofs = V.dofmap.cell_dofs(cell_id)`;
3. берет координаты dof через `V.tabulate_dof_coordinates()`;
4. вычисляет `local_rhs = gradients_p1_tetra(dof_coords[cell_dofs, :3]) @ moment`;
5. записывает четыре значения только в `cell_dofs`.

Для одного точечного диполя в P1 tetra RHS должен иметь ровно четыре ненулевых dof.

Диагностика:

```python
from sources import inspect_point_dipole_rhs_petsc

info = inspect_point_dipole_rhs_petsc(solver, source)
print(info["cell_id"])
print(info["cell_dofs"])
print(info["local_rhs"])
print(info["local_rhs_sum"])
```

`local_rhs_sum` должен быть близок к нулю, потому что сумма градиентов P1 базисных функций равна нулю. Для визуальной проверки RHS можно экспортировать его как VTX/BP:

```python
from forward import export_dolfinx_function_to_vtx
from sources import assemble_point_dipole_rhs_petsc

rhs = assemble_point_dipole_rhs_petsc(solver, source)
export_dolfinx_function_to_vtx(rhs, "output/rhs.bp", name="rhs")
```

## Debugging source location

`MeshData cell_id` и `DOLFINx cell_id` могут иметь разную индексацию, особенно после создания или распределения DOLFINx mesh.

- Для numpy RHS поле `source.cell_id` относится к `MeshData` ordering.
- Для PETSc/FEniCSx RHS явно переданный аргумент `cell_id` относится к DOLFINx ordering.
- По умолчанию `assemble_point_dipole_rhs_petsc(...)` не использует `source.cell_id`. Ячейка заново ищется по `source.position` через `locate_point_in_dolfinx_p1_tetra_mesh(...)`.
- `trust_source_cell_id=True` можно использовать только после явной проверки совпадения индексации.

Полная диагностика положения источника:

```python
from sources import inspect_point_dipole_location_petsc

info = inspect_point_dipole_location_petsc(solver, source)

print("declared position:", info["declared_position"])
print("declared cell id:", info["declared_cell_id"])
print("MeshData located cell id:", info["meshdata_located_cell_id"])
print("used DOLFINx cell id:", info["used_cell_id"])
print("cell dofs:", info["cell_dofs"])
print("barycentric:", info["barycentric_in_dolfinx_cell"])
print("inside:", info["is_inside_used_dolfinx_cell"])
print("dof cell center:", info["dof_cell_center"])
print("ordering warning:", info["ordering_warning"])
```

Проверка соответствия одинаковых integer cell ids:

```python
from sources import compare_meshdata_and_dolfinx_cell_centers

comparison = compare_meshdata_and_dolfinx_cell_centers(solver, max_cells=1000)
print(comparison["max_diff"])
print(comparison["worst_cell_id"])
```

Визуальный marker фактически используемой DOLFINx ячейки:

```python
from forward import export_dolfinx_function_to_vtx
from sources import create_cell_marker_function

marker = create_cell_marker_function(solver, info["used_cell_id"])
export_dolfinx_function_to_vtx(marker, "output/source_marker.bp", name="source_marker")
```

Для `torso.msh` диагностику можно выполнить так:

```python
from fem import NeumannPoissonSolver
from geometry import read_gmsh_meshio
from sources import PointDipole, inspect_point_dipole_location_petsc

tagged = read_gmsh_meshio("torso.msh", dim=3)
volume_mesh = tagged.to_mesh_data("tetra", physical_name="domain")
solver = NeumannPoissonSolver(volume_mesh)
source = PointDipole(position=desired_position, moment=[0.0, 0.0, 1.0])

info = inspect_point_dipole_location_petsc(solver, source)
assert info["is_inside_used_dolfinx_cell"]
assert abs(info["barycentric_sum"] - 1.0) < 1e-8
assert info["barycentric_min"] > -1e-8
```

Если `source_marker.bp` не совпадает с ожидаемым положением в ParaView, следует проверять `desired_position` и формирование source region. Знак RHS при этой диагностике не меняется: `local_rhs = grads @ source.moment`.
