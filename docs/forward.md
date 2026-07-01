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
- `nodal_values` — copy массива значений функции в ordering, указанном `nodal_value_ordering`;
- `dof_values` — явный accessor для DOLFINx dof-ordered values;
- `meshdata_nodal_values` — optional copy в MeshData node ordering для measurement evaluation;
- `raw_measurements` — значения до reference;
- `measurements` — referenced values;
- `reference` — выбранная reference-система;
- `metadata` — краткая информация о запуске.

Дополнительные properties: `num_nodes`, `num_electrodes`, `measurement_norm`, `raw_measurement_norm`, `has_meshdata_nodal_values`. `to_dict()` возвращает summary без больших массивов.

## Ordering note for measurements

`potential.x.array` использует DOLFINx DOF ordering, а numpy `MeasurementOperator` строится по MeshData node ordering. `ForwardSolver` применяет проверенное coordinate-based `node_id -> dof_id` mapping и переставляет значения перед вычислением измерений. Совпадение integer ids не считается универсальным свойством.

Mapping кэшируется на `NeumannPoissonSolver` и повторно используется между forward/Green solves. Не интерпретируйте `nodal_values[node_id]` как значение MeshData node без этого mapping.

Source RHS уже решает эту проблему отдельно: он собирается непосредственно через `V.dofmap.cell_dofs`.

## Green consistency

Модуль `green` решает reciprocal systems `K G_i = M_i^T` и строит `A[j, i, :] = grad G_i(x_j)`. Для candidate source ordinary forward measurement сравнивается с `A_j @ p` через `compare_forward_and_green`. Это одновременно проверяет RHS convention, node/DOF mapping и вычисление P1 gradients.

Small-mesh integration tests подтверждают текущий transfer sign `+1`. `GreenTransferMatrix.sign` остаётся явной частью API и учитывается через `matrix_for_candidate()`.

Подробности: [Green functions](green.md).

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

Source-cell markers are created in `sources` and exported through the same generic writer:

```python
from sources import create_cell_marker_function

marker = create_cell_marker_function(solver, dolfinx_cell_id, name="source_marker")
export_dolfinx_function_to_vtx(marker, "output/source_marker.bp", name="source_marker")
```

### Electrode marker export

For ParaView placement checks, `forward` can export electrodes as a diagnostic
P1 nodal marker field:

```python
from forward import (
    create_electrode_marker_function,
    export_electrode_markers_to_vtx,
    inspect_electrode_marker_mapping,
)

info = inspect_electrode_marker_mapping(solver, electrodes)
marker = create_electrode_marker_function(
    solver,
    electrodes,
    value_mode="index",
)

export_electrode_markers_to_vtx(
    solver,
    electrodes,
    "output/electrodes.bp",
    value_mode="index",
)
```

`value_mode="index"` writes marker values `1, 2, ...` for electrodes
`E1, E2, ...`; `value_mode="binary"` writes `1` for every electrode.
The marker is placed at the nearest FEM dof, not at an independent point-cloud
coordinate. Use `inspect_electrode_marker_mapping(...)` to inspect nearest dof
ids, distances and collisions.

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

## Forward convergence checks

Point dipole solution сингулярен в source position, поэтому global potential не обязан демонстрировать стандартную smooth P1 L2 convergence.

`test_forward_convergence.py` проверяет:

- RHS compatibility и localization на cell dofs;
- finite potential/measurements;
- average-reference zero sum;
- deterministic repeated solve;
- linearity по dipole moment;
- scaling по амплитуде момента;
- уменьшение differences между fixed remote observations при refinement `n=4, 8, 16`.

Для refinement source выбирается рядом с центром, но не на mesh face/edge/vertex. Диполь ровно в grid vertex принадлежит нескольким cells, и выбор одного local P1 gradient не образует однозначную refinement sequence.

Source lookup для repeated solves использует cached `DOLFINxP1TetraLocator`. Это ускоряет локализацию, но не устраняет математическую неоднозначность source на общей face/edge/vertex.

Запуск:

```bash
TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp RUN_DOLFINX_TESTS=1 \
  pytest test_forward_convergence.py
```
