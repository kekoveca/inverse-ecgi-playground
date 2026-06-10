# Geometry module

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

## Дальнейшие расширения

- `meshio` импорт `.msh`, `.vtk`, `.xdmf`;
- проекция электродов на surface mesh;
- spatial index через `scipy.spatial.cKDTree`;
- разметка областей торса;
- связь с `dolfinx.mesh.Mesh`;
- построение source region из heart mask / heart surface.

## Gmsh / meshio import with physical tags

The module now includes a tagged mesh container for meshes loaded through `meshio`:

```python
from geometry import read_gmsh_meshio

mesh = read_gmsh_meshio("torso.msh", dim=2)
print(mesh.cells.keys())
print(mesh.physical_tag("torso"))

domain = mesh.to_mesh_data("triangle", physical_name="torso")
```

Use `TaggedMesh` / `Mesh` when you need Gmsh physical groups, boundary tags, and several cell blocks. Use `MeshData` when a downstream component needs one concrete cell block, for example the 2D triangle domain or the 3D tetrahedral volume.


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
