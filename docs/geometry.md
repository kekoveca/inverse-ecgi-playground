# Geometry

Модуль `geometry` хранит и проверяет геометрические данные. Он не собирает FEM-матрицы и не зависит от DOLFINx.

## Main classes

- `MeshData` — единый контейнер сетки, cell blocks и physical tags.
- `ElectrodeSet` — позиции и labels электродов.
- `SourceRegion` — допустимые candidate points и соответствующие MeshData cell ids.
- `TorsoGeometry` — volume mesh, optional surface mesh, электроды и source region.
- `AffineTransform` — аффинные преобразования геометрических объектов.

Отдельные классы `TaggedMesh` и `Mesh` были удалены при объединении моделей. Их функции перенесены в `MeshData`; в текущем публичном API следует использовать только `MeshData`.

## MeshData

Основные поля:

```python
MeshData(
    points=...,       # (n_points, geometric_dim), float
    cells=...,        # active connectivity block
    cell_type="tetra",
    name="mesh",
    metadata={},
    cell_tags=None,
    field_data={},
    cell_blocks=None,
)
```

`points`, `cells` и `cell_type` задают активный block. При чтении Gmsh-сетки `cell_blocks` может одновременно содержать `tetra`, `triangle` и `line`.

Node ids и cell ids принадлежат только ordering этого `MeshData`. После преобразования в DOLFINx нельзя считать их равными DOF/cell ids.

## TaggedMesh compatibility

Исторический `TaggedMesh` больше не является отдельным классом. Gmsh/meshio import, `field_data`, `cell_tags`, multi-block storage, `physical_tag`, `physical_dimension`, `cell_block` и `to_mesh_data` реализованы непосредственно в `MeshData`.

Старый код и документацию следует переводить на `MeshData`; aliases `TaggedMesh`/`Mesh` не экспортируются.

## Gmsh physical tags

Внутренняя конвенция проекта:

```python
field_data: dict[str, tuple[int, int]]
# name -> (dim, tag)
```

Например:

```python
mesh.field_data["domain"] == (3, 1)
mesh.physical_dimension("domain") == 3
mesh.physical_tag("domain") == 1
```

Это Gmsh-подобный порядок `(dim, tag)`. `meshio` возвращает `name -> (tag, dim)`, поэтому `read_gmsh_meshio` переворачивает пары при импорте.

## Reading `.msh`

```python
from geometry import read_gmsh_meshio

tagged = read_gmsh_meshio("torso.msh", dim=3)

volume_mesh = tagged.to_mesh_data(
    cell_type="tetra",
    physical_name="domain",
)
surface_mesh = tagged.to_mesh_data(
    cell_type="triangle",
    physical_name="boundary",
)
```

`to_mesh_data` возвращает новый `MeshData` с одним активным block и сохраняет physical metadata.

## ElectrodeSet

```python
import numpy as np
from geometry import ElectrodeSet

electrodes = ElectrodeSet(
    positions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    labels=["E1", "E2"],
)
```

Для первичной проверки размещения доступен `electrode_placement_report(electrodes, mesh)`, который измеряет расстояния до ближайших mesh nodes.

Если электроды заданы немного вне volume mesh, модуль `measurements` умеет центрально проецировать их на surface mesh перед построением measurement operator. Эта операция не меняет исходный `ElectrodeSet` внутри `TorsoGeometry`; спроецированные позиции и report хранятся в `MeasurementOperator.metadata["electrode_projection"]`.

## SourceRegion

`SourceRegion` хранит:

```text
candidate_points
candidate_cell_ids  # MeshData cell ordering
```

Способы создания:

- `SourceRegion.all_cells(mesh)`;
- `SourceRegion.from_cell_ids(mesh, cell_ids)`;
- `SourceRegion.from_bounding_box(mesh, bounds_min, bounds_max, mode=...)`.

Bounding-box example:

```python
from geometry import SourceRegion

region = SourceRegion.from_bounding_box(
    mesh=volume_mesh,
    bounds_min=[-20.0, -20.0, -20.0],
    bounds_max=[20.0, 20.0, 20.0],
    mode="center",  # center, any_vertex, all_vertices
)
```

## TorsoGeometry

`TorsoGeometry` объединяет данные одного geometry case:

```python
from geometry import TorsoGeometry

torso = TorsoGeometry(
    geometry_id="case_001",
    volume_mesh=volume_mesh,
    surface_mesh=surface_mesh,
    electrodes=electrodes,
    source_region=region,
)
```

`validate_torso_geometry(torso)` проверяет согласованность dimensions, cell ids, electrode positions и наличие данных.

## Visualization

Визуализация предназначена для диагностики и требует `matplotlib`:

```python
import matplotlib.pyplot as plt
from geometry import plot_mesh, plot_source_region, plot_torso_geometry

plot_mesh(volume_mesh, max_cells=500, show_points=False)
plot_source_region(region)
plot_torso_geometry(
    torso,
    show_electrodes=True,
    show_source_region=True,
    max_cells=500,
    show_fig=True,
)
```

Для FEM fields и больших 3D-сеток используйте экспорт VTX/XDMF и ParaView.
