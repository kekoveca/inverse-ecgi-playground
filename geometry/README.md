# Geometry module

Full guide: [../docs/geometry.md](../docs/geometry.md).

`geometry` is the FEniCSx-independent data layer. It stores and validates:

- tetrahedral volume meshes and optional triangle surface meshes;
- electrode positions and labels;
- allowed source regions and candidate points;
- Gmsh physical groups and metadata;
- transforms and lightweight visualization data.

## Main types

- `MeshData`
- `ElectrodeSet`
- `SourceRegion`
- `TorsoGeometry`
- `AffineTransform`
- `GeometryValidationReport`

## Minimal example

```python
import numpy as np

from geometry import ElectrodeSet, MeshData, SourceRegion, TorsoGeometry, validate_torso_geometry

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

electrodes = ElectrodeSet(
    positions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    labels=["E1", "E2"],
)
source_region = SourceRegion.from_cell_ids(mesh, [0])
geometry = TorsoGeometry("demo", mesh, electrodes, source_region)

report = validate_torso_geometry(geometry)
print(report.is_valid)
```

## Gmsh import

```python
from geometry import read_gmsh_meshio

mesh = read_gmsh_meshio("meshes/torso.msh", dim=3)
volume_mesh = mesh.to_mesh_data("tetra", physical_name="domain")
surface_mesh = mesh.to_mesh_data("triangle", physical_name="boundary")
```

Internal `field_data` uses:

```text
name -> (dimension, physical_tag)
```

`meshio` returns `(tag, dimension)`, so `read_gmsh_meshio` reverses each pair during import.

## Source regions

```python
source_region = SourceRegion.from_bounding_box(
    volume_mesh,
    bounds_min=[-20.0, -20.0, -20.0],
    bounds_max=[20.0, 20.0, 20.0],
    mode="center",
)
```

`SourceRegion.candidate_cell_ids` are MeshData cell ids, not DOLFINx cell ids.

## Units and surface counts

All geometry objects must use one coordinate frame and unit system. No automatic mm/m conversion is performed.

An extracted triangle mesh may retain the full global point array. Count vertices actually used by the surface with:

```python
num_surface_vertices = np.unique(surface_mesh.cells.ravel()).size
```

Central electrode projection is implemented in `measurements`; DOLFINx conversion and lookup are implemented in `fem`.

## Visualization

Visualization imports `matplotlib` only when called:

```python
from geometry import plot_mesh, plot_source_region, plot_torso_geometry

plot_mesh(volume_mesh)
plot_source_region(source_region)
plot_torso_geometry(geometry, show_electrodes=True, show_source_region=True, show_fig=True)
```
