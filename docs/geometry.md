# Geometry

The `geometry` module stores and validates geometric data. It does not assemble FEM matrices and does not depend on DOLFINx.

## Main classes

- `MeshData` - the unified container for meshes, cell blocks, and physical tags.
- `ElectrodeSet` - electrode positions and labels.
- `SourceRegion` - allowed candidate points and corresponding MeshData cell ids.
- `TorsoGeometry` - volume mesh, optional surface mesh, electrodes, and source region.
- `AffineTransform` - affine transformations for geometry objects.

The separate `TaggedMesh` and `Mesh` classes were removed when the models were unified. Their functionality moved into `MeshData`; the current public API uses `MeshData` only.

## MeshData

Main fields:

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

`points`, `cells`, and `cell_type` define the active block. When reading a Gmsh mesh, `cell_blocks` may contain `tetra`, `triangle`, and `line` blocks at the same time.

Node and cell ids belong only to this `MeshData` ordering. After DOLFINx conversion, they must not be assumed equal to DOF or cell ids.

## TaggedMesh compatibility

The historical `TaggedMesh` is no longer a separate class. Gmsh/meshio import, `field_data`, `cell_tags`, multi-block storage, `physical_tag`, `physical_dimension`, `cell_block`, and `to_mesh_data` are implemented directly by `MeshData`.

Older code and documentation should use `MeshData`; `TaggedMesh` and `Mesh` aliases are not exported.

## Gmsh physical tags

Internal project convention:

```python
field_data: dict[str, tuple[int, int]]
# name -> (dim, tag)
```

For example:

```python
mesh.field_data["domain"] == (3, 1)
mesh.physical_dimension("domain") == 3
mesh.physical_tag("domain") == 1
```

This is the Gmsh-style `(dim, tag)` order. `meshio` returns `name -> (tag, dim)`, so `read_gmsh_meshio` reverses pairs during import.

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

`to_mesh_data` returns a new `MeshData` with one active block and preserves physical metadata.

Extracted surface blocks may retain the full global `points` array from the Gmsh file. Therefore `surface_mesh.num_points` can equal `volume_mesh.num_points` even when only a subset is referenced by boundary triangles. For diagnostics use:

```python
surface_used_vertex_ids = np.unique(surface_mesh.cells.ravel())
num_surface_used_vertices = surface_used_vertex_ids.size
```

## ElectrodeSet

```python
import numpy as np
from geometry import ElectrodeSet

electrodes = ElectrodeSet(
    positions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    labels=["E1", "E2"],
)
```

For an initial placement check, `electrode_placement_report(electrodes, mesh)` measures distances to the nearest mesh nodes.

If electrodes lie slightly outside the volume mesh, `measurements` can centrally project them onto the surface mesh before building the measurement operator. This does not mutate the original `ElectrodeSet` in `TorsoGeometry`; projected positions and the report are stored in `MeasurementOperator.metadata["electrode_projection"]`.

`ElectrodeProjectionReport.surface_cell_ids == -1` means no projection triangle was recorded for that electrode, usually because it was already inside/on the boundary and `project_only_outside=True`. It does not indicate a surface-mesh error.

## SourceRegion

`SourceRegion` stores:

```text
candidate_points
candidate_cell_ids  # MeshData cell ordering
```

Construction methods:

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

`TorsoGeometry` combines the data for one geometry case:

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

`validate_torso_geometry(torso)` checks dimensions, cell ids, electrode positions, and required data for consistency.

## Units and coordinate frame

`geometry` does not assign units automatically. Mesh points, electrodes, source regions, transforms, and distance thresholds must use one coordinate frame and unit system. If `.msh` uses millimeters, localization errors and projection distances are also in millimeters; `sigma` must be consistent with the physical model.

## Visualization

Visualization is intended for diagnostics and requires `matplotlib`:

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

Use VTX/XDMF export and ParaView for FEM fields and large 3D meshes.
