# Forward pipeline

## ForwardSolver

`ForwardSolver` связывает готовый Neumann solver, point dipole source и optional measurement operator:

```text
source
  |
  v
assemble RHS in DOLFINx ordering
  |
  v
solve potential
  |
  v
extract nodal values
  |
  v
evaluate measurements
  |
  v
ForwardResult
```

Stiffness matrix не пересобирается: `ForwardSolver` использует уже созданный `NeumannPoissonSolver`.

```python
from forward import ForwardSolver

forward = ForwardSolver(
    poisson_solver=solver,
    electrodes=electrodes,
    reference="average",
)
result = forward.solve(source)
```

Если electrodes и measurement operator не переданы, solver всё равно возвращает potential, а measurement arrays будут пустыми.

## ForwardResult

Поля:

- `source` — исходный `PointDipole`;
- `potential` — обычно `dolfinx.fem.Function`;
- `nodal_values` — copy массива значений функции;
- `raw_measurements` — значения до reference;
- `measurements` — referenced values;
- `reference` — выбранная reference-система;
- `metadata` — краткая информация о запуске.

Дополнительные properties: `num_nodes`, `num_electrodes`, `measurement_norm`, `raw_measurement_norm`. `to_dict()` возвращает summary без больших массивов.

## Ordering note for measurements

`potential.x.array` использует DOLFINx DOF ordering, а numpy `MeasurementOperator` строится по MeshData node ordering. Для production-моделей следует использовать только проверенное node-to-dof mapping. Нельзя считать совпадение ordering универсальным свойством.

Source RHS уже решает эту проблему отдельно: он собирается непосредственно через `V.dofmap.cell_dofs`.

## Export

### VTX/BP

Предпочтительный формат ParaView:

```python
from forward import export_forward_result_to_vtx

export_forward_result_to_vtx(result, "output/potential.bp")
```

Откройте `.bp` output в ParaView.

Generic DOLFINx Function export:

```python
from forward import export_dolfinx_function_to_vtx

export_dolfinx_function_to_vtx(rhs, "output/rhs.bp", name="rhs")
```

### XDMF

```python
from forward import export_forward_result_to_xdmf

export_forward_result_to_xdmf(result, "output/potential.xdmf")
```

Откройте `.xdmf` в ParaView. Рядом создаётся `.h5`; эти файлы должны оставаться вместе. Если ParaView падает или показывает пустой XDMF, используйте VTX/BP.

## Full example

```python
from geometry import ElectrodeSet, read_gmsh_meshio
from fem import NeumannPoissonSolver
from sources import PointDipole
from forward import ForwardSolver, export_forward_result_to_vtx

tagged = read_gmsh_meshio("torso.msh", dim=3)
mesh = tagged.to_mesh_data("tetra", physical_name="domain")
electrodes = ElectrodeSet(
    positions=mesh.points[[0, 1]].copy(),
    labels=["E1", "E2"],
)

solver = NeumannPoissonSolver(mesh, degree=1, sigma=1.0)
try:
    pipeline = ForwardSolver(solver, electrodes=electrodes, reference="average")
    source = PointDipole(position=[0.0, 0.0, 0.0], moment=[0.0, 0.0, 1.0])
    result = pipeline.solve(source)
    export_forward_result_to_vtx(result, "output/potential.bp")
finally:
    solver.destroy()
```
