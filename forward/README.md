# forward

Полная документация: [../docs/forward.md](../docs/forward.md).

`forward` собирает прямой pipeline для точечного диполя:

```text
PointDipole(x0, p)
    -> assemble RHS
    -> solve Neumann Poisson FEM
    -> extract nodal potential
    -> measure electrodes
    -> ForwardResult
```

Модуль не собирает FEM-матрицу сам. Он использует готовый `fem.NeumannPoissonSolver`, RHS из `sources` и оператор измерений из `measurements`.

## Минимальный пример

```python
from geometry import ElectrodeSet, read_gmsh_meshio
from fem import NeumannPoissonSolver
from sources import PointDipole
from forward import ForwardSolver, export_forward_result_to_vtx, export_forward_result_to_xdmf

tagged = read_gmsh_meshio("torso.msh", dim=3)

volume_mesh = tagged.to_mesh_data(
    cell_type="tetra",
    physical_name="domain",
)

solver = NeumannPoissonSolver(
    mesh=volume_mesh,
    degree=1,
    sigma=1.0,
)

electrodes = ElectrodeSet(
    positions=...,  # shape (n_electrodes, 3)
    labels=...,
)

forward = ForwardSolver(
    poisson_solver=solver,
    electrodes=electrodes,
    reference="average",
)

source = PointDipole(
    position=[...],
    moment=[1.0, 0.0, 0.0],
)

result = forward.solve(source)
print(result.measurements)

export_forward_result_to_xdmf(
    result,
    "output/potential.xdmf",
)

export_forward_result_to_vtx(
    result,
    "output/potential.bp",
)
```

XDMF export может создать два файла: `.xdmf` и `.h5`. Открывать в ParaView нужно файл `.xdmf`.

Если ParaView падает при открытии XDMF или показывает пустой результат, используйте VTX/BP export:

```python
from forward import export_forward_result_to_vtx

export_forward_result_to_vtx(result, "output/potential.bp")
```

Для VTX export открывайте в ParaView `.bp` output.

Для проверки размещения электродов можно экспортировать диагностический marker field:

```python
from forward import export_electrode_markers_to_vtx

export_electrode_markers_to_vtx(solver, electrodes, "output/electrodes.bp")
```

`electrodes.bp` отмечает ближайшие FEM DOF к координатам электродов; это не отдельное point-cloud представление.

Для чистой задачи Неймана потенциал определен с точностью до константы. `fem` фиксирует gauge решения, а `average` reference дополнительно убирает произвольную константу из электродных измерений.

Знак RHS задается в `sources` как `gradients_p1_tetra(vertices) @ moment` и будет окончательно проверяться через Green consistency.
