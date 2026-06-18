# Cardio direct/inverse benchmark

## Project goal

Проект реализует forward pipeline для вычисления электрического потенциала в объёмной сетке торса от точечного диполя:

```text
geometry -> FEM Neumann solver -> point dipole RHS -> electrode measurements -> ParaView export
```

Текущий код является основой для будущего benchmark по Green-функциям и обратной задаче. Green/inverse-модули пока не входят в проект.

## Current modules

- [`geometry`](docs/geometry.md) — независимые от FEniCSx сетки, physical groups, электроды, source region и геометрические проверки.
- [`fem`](docs/fem.md) — DOLFINx/PETSc solver задачи Пуассона с чистыми условиями Неймана.
- [`sources`](docs/sources.md) — геометрия P1 tetra и RHS точечного диполя в numpy и FEniCSx DOF ordering.
- [`measurements`](docs/measurements.md) — интерполяция потенциала на электроды и reference-системы.
- [`forward`](docs/forward.md) — полный pipeline `source -> rhs -> potential -> measurements -> result` и экспорт в ParaView.

Вспомогательный модуль [`verification`](verification/README.md) содержит unit-cube mesh refinement, manufactured solutions и convergence reports.

## Important conventions

- `MeshData` хранит node/cell ids в собственном ordering. Они не обязаны совпадать с DOLFINx DOF/cell ids.
- Внутренний `field_data` имеет Gmsh-подобный формат `name -> (dim, tag)`. `meshio` возвращает `(tag, dim)`, поэтому импорт переворачивает пары.
- Чистая задача Неймана имеет константное ядро. Solver использует PETSc `NullSpace` и фиксирует gauge решения.
- Для точечного диполя используется `local_rhs = gradients_p1_tetra(vertices) @ moment`. Физический знак будет отдельно проверен через FEM/Green consistency.
- Измерения задаются как `y_raw = P u`, `g = R P u`; стандартный reference — `average`.
- Для ParaView предпочтителен VTX/BP. XDMF остаётся доступным как альтернативный формат.

Подробности и диагностические процедуры собраны в [architecture](docs/architecture.md) и [debugging](docs/debugging.md).

## Minimal example

```python
from geometry import ElectrodeSet, read_gmsh_meshio
from fem import NeumannPoissonSolver
from sources import PointDipole
from forward import ForwardSolver, export_forward_result_to_vtx

tagged = read_gmsh_meshio("torso.msh", dim=3)
volume_mesh = tagged.to_mesh_data(
    cell_type="tetra",
    physical_name="domain",
)

electrodes = ElectrodeSet(
    positions=volume_mesh.points[[0, 1]].copy(),
    labels=["E1", "E2"],
)

solver = NeumannPoissonSolver(
    mesh=volume_mesh,
    degree=1,
    sigma=1.0,
)

try:
    forward = ForwardSolver(
        poisson_solver=solver,
        electrodes=electrodes,
        reference="average",
    )
    source = PointDipole(
        position=[0.0, 0.0, 0.0],
        moment=[0.0, 0.0, 1.0],
    )
    result = forward.solve(source)
    export_forward_result_to_vtx(result, "output/potential.bp")
finally:
    solver.destroy()
```

Откройте `output/potential.bp` в ParaView.

## Running the included example

`main.py` решает задачу на `torso.msh` и экспортирует potential, RHS и marker ячейки источника:

```bash
python3 main.py --position 0 0 0 --moment 0 0 1
```

Результаты записываются в `output/`.

## Testing

Основной набор тестов:

```bash
pytest
```

Numpy/scipy-тесты работают без DOLFINx. Реальные DOLFINx/MPI-тесты защищены переменной окружения и без неё будут skipped даже при установленном DOLFINx:

```bash
TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp RUN_DOLFINX_TESTS=1 pytest
```

Численные проверки forward/FEM:

```bash
TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp RUN_DOLFINX_TESTS=1 \
  pytest test_forward_convergence.py test_poisson_manufactured_solution.py
```

Точечный диполь сингулярен, поэтому для него не требуется классическая глобальная L2-сходимость potential. Forward test проверяет стабилизацию average-referenced measurements вдали от источника. L2 convergence FEM solver отдельно проверяется на smooth manufactured cosine solution.

## Documentation

- [Architecture](docs/architecture.md)
- [Geometry](docs/geometry.md)
- [FEM](docs/fem.md)
- [Sources](docs/sources.md)
- [Measurements](docs/measurements.md)
- [Forward pipeline](docs/forward.md)
- [Debugging](docs/debugging.md)
- [Examples](docs/examples.md)
- [MeshData details](docs/mesh_data.md)
- [Verification utilities](verification/README.md)
