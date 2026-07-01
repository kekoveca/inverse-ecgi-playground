# MeshData: Gmsh/meshio meshes with physical groups

`geometry.mesh_model.MeshData` stores meshes read from Gmsh through `meshio` together with physical groups. This layer is useful when one file contains different cell blocks, such as `tetra` for the torso volume and `triangle` for its surface.

Downstream code that needs one specific block converts a multi-block `MeshData` into a single-block `MeshData` through `to_mesh_data(...)`.

## Main entities

- `MeshData`: unified container for coordinates, cell blocks, physical tags, and metadata.
- `read_gmsh_meshio(path, dim)`: reads `.msh` through `meshio`.
- `_field_data_to_tuples(field_data)`: internal conversion of `meshio.field_data`.

## Internal `field_data` convention

Within the project, `field_data` always uses Gmsh API ordering:

```python
field_data: dict[str, tuple[int, int]]
# name -> (dim, tag)
```

Examples:

```python
field_data["domain"] == (3, 1)    # 3D volume group, tag=1
field_data["boundary"] == (2, 2)  # 2D surface group, tag=2
```

This convention matters: the first pair element is the physical group's geometric dimension, and the second is the physical tag.

## Difference from `meshio`

`meshio` usually returns `field_data` in the opposite order:

```text
# meshio convention
name -> (tag, dim)
```

Therefore `read_gmsh_meshio()` calls `_field_data_to_tuples()` and reverses each pair:

```text
meshio:   (tag, dim)
internal: (dim, tag)
```

For example:

```python
from geometry.mesh_model import _field_data_to_tuples
import numpy as np

converted = _field_data_to_tuples(
    {
        "domain": np.array([1, 3]),
        "boundary": np.array([2, 2]),
    }
)

assert converted["domain"] == (3, 1)
assert converted["boundary"] == (2, 2)
```

When creating `MeshData` manually, provide the internal `(dim, tag)` order.

## Multi-block `MeshData` structure

```python
from geometry import MeshData
import numpy as np

mesh = MeshData.from_cell_blocks(
    points=np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    cell_blocks={
        "tetra": np.array([[0, 1, 2, 3]]),
    },
    cell_tags={
        "tetra": np.array([1]),
    },
    field_data={
        "domain": (3, 1),
    },
)
```

Fields:

- `dim`: working coordinate dimension, usually `2` or `3`.
- `points`: node array with shape `(n_nodes, dim)`.
- `cells`: active-block connectivity.
- `cell_type`: active block type.
- `cell_blocks`: `cell_type -> connectivity`, such as `"tetra"`, `"triangle"`, or `"line"`.
- `cell_tags`: `cell_type -> physical tag per cell`.
- `field_data`: `physical_name -> (dim, tag)`.
- `metadata`: arbitrary metadata.

If a cell block has no `cell_tags`, the container creates zero tags for all cells in that block.

Compatibility aliases:

- `mesh.coords == mesh.points`;
- `mesh.dim == mesh.geometric_dim`.

## Physical group API

```python
assert mesh.physical_dimension("domain") == 3
assert mesh.physical_tag("domain") == 1
```

`physical_dimension(name)` returns the first element of `field_data[name]`.

`physical_tag(name)` returns the second element of `field_data[name]`.

For an unknown physical group, these methods raise `KeyError` with the available groups.

## Filtering cell blocks

`cell_block(cell_type, physical_name=None)` returns the connectivity array for a cell type. If `physical_name` is supplied, cells are filtered by physical tag:

```python
domain_cells = mesh.cell_block("tetra", physical_name="domain")
```

Filtering is by tag, not by dimension.

Equivalent logic:

```python
tag = mesh.physical_tag("domain")
tags = mesh.cell_tags["tetra"]
domain_cells = mesh.cell_blocks["tetra"][tags == tag]
```

## Conversion to single-block `MeshData`

`to_mesh_data()` converts one cell block into a lightweight `MeshData` container.

```python
domain = mesh.to_mesh_data(
    cell_type="tetra",
    physical_name="domain",
)

assert domain.cell_type == "tetra"
assert domain.metadata["physical_name"] == "domain"
assert domain.metadata["physical_dimension"] == 3
assert domain.metadata["physical_tag"] == 1
```

If `physical_name` is omitted, the result contains the entire selected `cell_type` block.

Result metadata contains:

- `source`: `"MeshData"`;
- `cell_type`: selected cell type;
- `physical_name`: group name or `None`;
- `field_data`: internal physical-group dictionary;
- `physical_dimension`: present only when `physical_name` is supplied;
- `physical_tag`: present only when `physical_name` is supplied.

## Reading `.msh`

```python
from geometry import read_gmsh_meshio

mesh = read_gmsh_meshio("torso.msh", dim=3)

print(mesh.field_data)
print(mesh.physical_dimension("domain"))
print(mesh.physical_tag("domain"))
```

For a mesh with the standard example tags:

```python
mesh.field_data["domain"] == (3, 1)
mesh.field_data["boundary"] == (2, 2)
mesh.physical_dimension("domain") == 3
mesh.physical_tag("domain") == 1
```

Extracting volume and surface:

```python
volume_mesh = mesh.to_mesh_data(
    cell_type="tetra",
    physical_name="domain",
)

surface_mesh = mesh.to_mesh_data(
    cell_type="triangle",
    physical_name="boundary",
)
```

Do not tie production code to exact counts in repository mesh files; they change under refinement and remeshing.

### Surface point arrays

`to_mesh_data("triangle", ...)` preserves the original global point array and selected triangle connectivity. Therefore `surface_mesh.num_points` may equal `volume_mesh.num_points` even though surface triangles use substantially fewer vertices:

```python
used_surface_vertex_ids = np.unique(surface_mesh.cells.ravel())
print("point array size:", surface_mesh.num_points)
print("used surface vertices:", used_surface_vertex_ids.size)
```

This representation preserves global node ids and is not an error. An explicit compaction helper could reduce memory, but nodes must not be renumbered implicitly when downstream metadata depends on original ids.

## Typical pipeline

```python
import numpy as np

from geometry import (
    ElectrodeSet,
    SourceRegion,
    TorsoGeometry,
    read_gmsh_meshio,
    validate_torso_geometry,
)

mesh = read_gmsh_meshio("torso.msh", dim=3)

volume_mesh = mesh.to_mesh_data("tetra", physical_name="domain")
surface_mesh = mesh.to_mesh_data("triangle", physical_name="boundary")

electrodes = ElectrodeSet(
    positions=np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    ),
    labels=["E1", "E2"],
)

source_region = SourceRegion.from_bounding_box(
    mesh=volume_mesh,
    bounds_min=[-50.0, -50.0, -50.0],
    bounds_max=[50.0, 50.0, 50.0],
    mode="center",
)

geometry = TorsoGeometry(
    geometry_id="torso_from_msh",
    volume_mesh=volume_mesh,
    surface_mesh=surface_mesh,
    electrodes=electrodes,
    source_region=source_region,
)

report = validate_torso_geometry(geometry)
print(report.is_valid)
print(report.summary)
```

## Common mistakes

If `cell_block("tetra", physical_name="domain")` returns zero cells, verify:

- `field_data["domain"]` uses `(dim, tag)`, for example `(3, 1)`;
- `cell_tags["tetra"]` contains the physical tag, for example `1`;
- data came through `read_gmsh_meshio()` when the source was `meshio`;
- you did not manually pass meshio's `(tag, dim)` order into `MeshData`.

To use the full mesh without physical-group filtering:

```python
all_tetra = mesh.cell_block("tetra")
all_volume = mesh.to_mesh_data("tetra")
```
