# Measurements module

Full guide: [../docs/measurements.md](../docs/measurements.md).

`measurements` converts a MeshData-node-ordered P1 potential into raw and referenced electrode values. Its core implementation uses numpy/scipy and does not require DOLFINx.

## Operators

```text
y_raw = P u
g = R y_raw
M = R P
```

Each P1 tetra interpolation row has four barycentric weights.

```python
from measurements import build_measurement_operator

operator = build_measurement_operator(
    mesh=volume_mesh,
    electrodes=electrodes,
    reference="average",
    sparse=True,
    surface_mesh=surface_mesh,
)

raw = operator.evaluate_raw(nodal_values)
referenced = operator.evaluate(nodal_values)
P = operator.raw_matrix()
M = operator.matrix()
```

Supported references:

- `none`: unchanged values;
- `average`: subtract the electrode mean;
- `single`: subtract the electrode selected by `reference_index`.

For pure-Neumann Green solves, measurement rows must sum to zero. Average and valid single references satisfy this condition; none generally does not.

## Outside electrodes

`build_measurement_operator` can centrally project electrodes outside the tetra volume onto a supplied triangle surface or an inferred tetra boundary.

```python
from measurements import central_project_electrodes_to_surface

projected, report = central_project_electrodes_to_surface(
    volume_mesh,
    electrodes,
    surface_mesh=surface_mesh,
)
```

`TetraVolumeLocator` caches volume geometry and a KD-tree for repeated inside checks. `CentralSurfaceProjector` caches triangle geometry for central ray projection. Pass reusable instances explicitly when projecting several electrode sets on one geometry.

`report.surface_cell_ids[i] == -1` normally means electrode `i` was left unchanged by `project_only_outside=True`; inspect `projected_mask` as well.

Measurement matrices remain in MeshData node ordering after projection. Forward and Green adapters apply the DOLFINx node-to-DOF mapping.
