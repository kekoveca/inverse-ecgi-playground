# Measurements

## Purpose

`measurements` converts a nodal P1 potential into electrode values and applies a reference system.

```text
nodal values u -> raw electrode values y_raw -> referenced values g
```

The core implementation uses numpy/scipy and does not require DOLFINx.

## Point location

- `locate_points_in_tetra_mesh(mesh, points)` returns MeshData cell ids and barycentric coordinates.
- `locate_electrodes_in_mesh(mesh, electrodes)` applies the same algorithm to `ElectrodeSet.positions`.
- `central_project_electrodes_to_surface(...)` centrally projects outside electrodes onto a surface mesh or a boundary extracted from the tetra volume mesh.
- `TetraVolumeLocator` and `CentralSurfaceProjector` cache spatial data for repeated inside checks and central ray projection.

Tetrahedron geometry is reused from `sources`; `measurements` does not duplicate barycentric mathematics.

Returned cell ids use MeshData ordering.

`locate_points_in_tetra_mesh` remains strict and raises `ValueError` for points outside the volume mesh. Electrode handling is more permissive: by default, `build_measurement_operator` centrally projects electrodes that are outside the torso. Projection casts a ray from the volume center through the outside electrode and takes the first surface intersection. If `surface_mesh` is omitted, boundary triangles are extracted from the tetra mesh.

```python
from measurements import central_project_electrodes_to_surface

projected_electrodes, report = central_project_electrodes_to_surface(
    volume_mesh=volume_mesh,
    electrodes=electrodes,
    surface_mesh=surface_mesh,  # optional
)

print(report.projected_indices)
print(report.max_projection_distance)
```

For large meshes, projection uses locator/projector objects inside
`central_project_electrodes_to_surface` so it does not rebuild a volume-cell
KD-tree for every electrode. For multiple projection runs on one geometry,
create and pass these objects explicitly:

```python
from measurements import CentralSurfaceProjector, TetraVolumeLocator

volume_locator = TetraVolumeLocator(volume_mesh)
surface_projector = CentralSurfaceProjector(surface_mesh, center=volume_mesh.points.mean(axis=0))

projected_electrodes, report = central_project_electrodes_to_surface(
    volume_mesh,
    electrodes,
    surface_mesh=surface_mesh,
    volume_locator=volume_locator,
    surface_projector=surface_projector,
)
```

`TetraVolumeLocator` accelerates inside checks with cached tetra geometry, centroids, and a KD-tree. `CentralSurfaceProjector` caches the triangle array, but the current ray intersection still checks surface triangles for every electrode that is actually projected. Reuse both objects across runs on one geometry.

In `ElectrodeProjectionReport`:

- `projected_mask[i]` shows whether electrode `i` moved;
- `surface_cell_ids[i]` is set only for projected electrodes;
- `-1` usually means “left unchanged,” not a surface-search failure;
- `projection_distances` and `max_projection_distance` should be part of real-experiment quality control.

## Interpolation matrix

For P1 tetrahedra:

```text
y_raw = P u
```

Each row of `P` contains four barycentric weights in the columns of the containing tetrahedron's nodes.

```python
from measurements import build_point_interpolation_matrix

P = build_point_interpolation_matrix(
    mesh=volume_mesh,
    points=electrodes.positions,
    sparse=True,
)
```

With `sparse=True`, the function returns a scipy CSR matrix. If scipy is unavailable, it warns and returns a dense array.

For this matrix, `u` must use **MeshData node ordering**. DOLFINx DOF ordering must not be assumed identical without a verified mapping.

## Reference systems

Supported references:

- `none`: `g = y`;
- `average`: `g = y - mean(y)`;
- `single`: `g_i = y_i - y_reference_index`.

```python
from measurements import apply_reference

g_average = apply_reference(y_raw, reference="average")
g_single = apply_reference(y_raw, reference="single", reference_index=0)
```

The reference matrix is denoted by `R`.

## MeasurementOperator

The full linear operator is:

```text
M = R @ P
g = M u
```

```python
from measurements import build_measurement_operator

op = build_measurement_operator(
    mesh=volume_mesh,
    electrodes=electrodes,
    reference="average",
    sparse=True,
    surface_mesh=surface_mesh,  # optional
)

y_raw = op.evaluate_raw(nodal_values)
g = op.evaluate(nodal_values)
P = op.raw_matrix()
M = op.matrix()
```

Methods:

- `raw_matrix()` — `P`;
- `matrix()` — `M = R @ P`;
- `evaluate_raw()` — raw electrode values;
- `evaluate()` — referenced values.

Rows of `M` are used as RHS vectors in `green`: `K G_i = M_i^T`. Before writing them to a DOLFINx Function, they are mapped from MeshData node ordering to DOLFINx DOF ordering.

If electrodes were projected, the summary is available in `op.metadata["electrode_projection"]`.

Measurement matrices always use MeshData node ordering, regardless of whether positions were projected. Reordering into DOLFINx DOF ordering happens in `forward`/`green`, not inside `MeasurementOperator`.

### Green RHS compatibility

For pure-Neumann Green solves every row of `M` must sum to zero. `average` and valid `single` references satisfy this for P1 interpolation; `none` generally does not. The `green` module checks row sums before solving and raises on incompatible measurement functionals.

## Constant potential test

Average reference must eliminate a constant:

```python
u_constant = np.full(volume_mesh.num_points, 5.0)
g = op.evaluate(u_constant)
assert np.allclose(g, 0.0)
```

This matches the invariance of electrode measurements to the pure-Neumann gauge.
