# Geometry module

Полная документация: [../docs/geometry.md](../docs/geometry.md).

Независимый слой геометрии для бенчмарка геометрий торса.

## Назначение

Модуль подготавливает данные для FEM/FEniCSx-слоя:

- объёмную сетку торса;
- surface mesh, если есть;
- набор электродов;
- область допустимых источников;
- candidate source points;
- метаданные и базовые проверки качества.

Модуль не собирает FEM-матрицы и не зависит от FEniCSx.

## Основные сущности

- `MeshData`
- `ElectrodeSet`
- `SourceRegion`
- `TorsoGeometry`
- `AffineTransform`
- `GeometryValidationReport`

## Минимальный пример

```python
import numpy as np
from geometry import MeshData, ElectrodeSet, SourceRegion, TorsoGeometry, validate_torso_geometry

points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
], dtype=float)

cells = np.array([[0, 1, 2, 3]], dtype=np.int64)

mesh = MeshData(points=points, cells=cells, cell_type="tetra", name="single_tet")

electrodes = ElectrodeSet(
    positions=np.array([[1, 0, 0], [0, 1, 0]], dtype=float),
    labels=["E1", "E2"],
)

source_region = SourceRegion.from_cell_ids(mesh, np.array([0], dtype=np.int64))

geometry = TorsoGeometry(
    geometry_id="demo",
    volume_mesh=mesh,
    electrodes=electrodes,
    source_region=source_region,
)

report = validate_torso_geometry(geometry)
print(report.is_valid)
print(report.summary)
```

## Связанные слои и будущие расширения

- Gmsh/meshio import и physical groups уже поддерживаются в `MeshData`;
- central electrode projection и cached volume locator находятся в `measurements`;
- DOLFINx conversion, node↔DOF mapping и local-cell locator находятся в `fem`;
- возможные расширения: heart mask/surface source regions, explicit surface compaction и registration-aware transforms.

## Gmsh / meshio import with physical tags

`MeshData` поддерживает сетки, загруженные через `meshio` из Gmsh-файлов с physical groups. Один и тот же объект может хранить несколько cell blocks, например `tetra` для объема и `triangle` для поверхности. Подробная документация: [../docs/mesh_data.md](../docs/mesh_data.md).

Внутренняя конвенция проекта для `field_data`:

```python
field_data: dict[str, tuple[int, int]]
# name -> (dim, tag)
```

Например, для 3D-объема `domain` с tag `1`:

```python
mesh.field_data["domain"] == (3, 1)
```

`meshio` обычно возвращает пары в порядке `(tag, dim)`, поэтому `read_gmsh_meshio()` при чтении `.msh` переворачивает их во внутренний формат `(dim, tag)`.

Минимальный пример:

```python
from geometry import read_gmsh_meshio

mesh = read_gmsh_meshio("torso.msh", dim=3)

print(mesh.cell_blocks.keys())
print(mesh.field_data)
print(mesh.physical_dimension("domain"))
print(mesh.physical_tag("domain"))

volume_mesh = mesh.to_mesh_data("tetra", physical_name="domain")
surface_mesh = mesh.to_mesh_data("triangle", physical_name="boundary")
```

Используйте multi-block `MeshData`, когда нужны Gmsh physical groups, boundary tags и несколько cell blocks в одном объекте. Используйте `to_mesh_data(...)`, когда downstream-компоненту нужен один конкретный блок: например 3D tetrahedral volume или surface mesh из triangles.


## Source region from bounding box

For the first inverse-problem prototype, candidate dipole locations can be selected by an axis-aligned bounding box. Candidate points are cell centers of selected cells.

```python
from geometry import SourceRegion

source_region = SourceRegion.from_bounding_box(
    mesh=volume_mesh,
    bounds_min=[-0.2, -0.2, 0.0],
    bounds_max=[0.2, 0.2, 0.5],
    mode="center",  # "center", "any_vertex", or "all_vertices"
)
```

## Geometry visualization

Visualization is optional and only requires `matplotlib` when plotting functions are called.

```python
from geometry import plot_torso_geometry

fig, ax = plot_torso_geometry(
    geometry,
    show_electrodes=True,
    show_source_region=True,
    max_cells=500,
)
```

For lower-level diagnostics:

```python
from geometry import plot_mesh, plot_source_region

plot_mesh(volume_mesh)
plot_source_region(source_region)
```

## Units and surface counts

Все geometry objects должны использовать одну coordinate frame и единицы; автоматической конверсии mm/m нет. Extracted triangle meshes могут хранить полный global point array. Для фактического числа surface vertices используйте `np.unique(surface_mesh.cells.ravel()).size`, а не только `surface_mesh.num_points`.
