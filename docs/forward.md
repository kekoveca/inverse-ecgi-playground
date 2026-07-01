# Forward pipeline

## ForwardSolver

`ForwardSolver` connects an existing Neumann solver, point-dipole source, and optional measurement operator:

```text
source
  |
  v
assemble RHS in DOLFINx ordering
  |
  v
solve potential
  |
  v
extract nodal values
  |
  v
evaluate measurements
  |
  v
ForwardResult
```

The stiffness matrix is not reassembled: `ForwardSolver` uses an existing `NeumannPoissonSolver`.

```python
from forward import ForwardSolver

forward = ForwardSolver(
    poisson_solver=solver,
    electrodes=electrodes,
    reference="average",
)
result = forward.solve(source)
```

If neither electrodes nor a measurement operator are supplied, the solver still returns the potential and the measurement arrays are empty.

## ForwardResult

Fields:

- `source` - the original `PointDipole`;
- `potential` - usually a `dolfinx.fem.Function`;
- `nodal_values` - a copy of function values in the ordering specified by `nodal_value_ordering`;
- `dof_values` - an explicit accessor for DOLFINx DOF-ordered values;
- `meshdata_nodal_values` - an optional copy in MeshData node ordering for measurement evaluation;
- `raw_measurements` - values before referencing;
- `measurements` — referenced values;
- `reference` - the selected reference system;
- `metadata` - lightweight run information.

Additional properties are `num_nodes`, `num_electrodes`, `measurement_norm`, `raw_measurement_norm`, and `has_meshdata_nodal_values`. `to_dict()` returns a summary without large arrays.

## Ordering note for measurements

`potential.x.array` uses DOLFINx DOF ordering, while the numpy `MeasurementOperator` is built in MeshData node ordering. `ForwardSolver` applies a verified coordinate-based `node_id -> dof_id` mapping and reorders values before evaluating measurements. Equal integer ids are not assumed to be universal.

The mapping is cached on `NeumannPoissonSolver` and reused between forward/Green solves. Do not interpret `nodal_values[node_id]` as a MeshData-node value without this mapping.

The source RHS handles this boundary separately by assembling directly through `V.dofmap.cell_dofs`.

## Green consistency

The `green` module solves reciprocal systems `K G_i = M_i^T` and builds `A[j, i, :] = grad G_i(x_j)`. For a candidate source, `compare_forward_and_green` compares ordinary forward measurements with `A_j @ p`. This simultaneously checks the RHS convention, node/DOF mapping, and P1 gradient calculation.

Small-mesh integration tests confirm transfer sign `+1`. `GreenTransferMatrix.sign` remains explicit in the API and is applied by `matrix_for_candidate()`.

See [Green functions](green.md) for details.

## Export

### VTX/BP

Preferred ParaView format:

```python
from forward import export_forward_result_to_vtx

export_forward_result_to_vtx(result, "output/potential.bp")
```

Open the `.bp` output in ParaView.

Generic DOLFINx Function export:

```python
from forward import export_dolfinx_function_to_vtx

export_dolfinx_function_to_vtx(rhs, "output/rhs.bp", name="rhs")
```

Source-cell markers are created in `sources` and exported through the same generic writer:

```python
from sources import create_cell_marker_function

marker = create_cell_marker_function(solver, dolfinx_cell_id, name="source_marker")
export_dolfinx_function_to_vtx(marker, "output/source_marker.bp", name="source_marker")
```

### Electrode marker export

For ParaView placement checks, `forward` can export electrodes as a diagnostic
P1 nodal marker field:

```python
from forward import (
    create_electrode_marker_function,
    export_electrode_markers_to_vtx,
    inspect_electrode_marker_mapping,
)

info = inspect_electrode_marker_mapping(solver, electrodes)
marker = create_electrode_marker_function(
    solver,
    electrodes,
    value_mode="index",
)

export_electrode_markers_to_vtx(
    solver,
    electrodes,
    "output/electrodes.bp",
    value_mode="index",
)
```

`value_mode="index"` writes marker values `1, 2, ...` for electrodes
`E1, E2, ...`; `value_mode="binary"` writes `1` for every electrode.
The marker is placed at the nearest FEM dof, not at an independent point-cloud
coordinate. Use `inspect_electrode_marker_mapping(...)` to inspect nearest dof
ids, distances and collisions.

### XDMF

```python
from forward import export_forward_result_to_xdmf

export_forward_result_to_xdmf(result, "output/potential.xdmf")
```

Open the `.xdmf` file in ParaView. A companion `.h5` file is created and must remain beside it. If ParaView crashes or shows an empty XDMF, use VTX/BP.

## Full example

```python
from geometry import ElectrodeSet, read_gmsh_meshio
from fem import NeumannPoissonSolver
from sources import PointDipole
from forward import ForwardSolver, export_forward_result_to_vtx

tagged = read_gmsh_meshio("torso.msh", dim=3)
mesh = tagged.to_mesh_data("tetra", physical_name="domain")
electrodes = ElectrodeSet(
    positions=mesh.points[[0, 1]].copy(),
    labels=["E1", "E2"],
)

solver = NeumannPoissonSolver(mesh, degree=1, sigma=1.0)
try:
    pipeline = ForwardSolver(solver, electrodes=electrodes, reference="average")
    source = PointDipole(position=[0.0, 0.0, 0.0], moment=[0.0, 0.0, 1.0])
    result = pipeline.solve(source)
    export_forward_result_to_vtx(result, "output/potential.bp")
finally:
    solver.destroy()
```

## Forward convergence checks

The point-dipole solution is singular at the source position, so the global potential is not expected to exhibit standard smooth P1 L2 convergence.

`test_forward_convergence.py` checks:

- RHS compatibility and localization on cell DOFs;
- finite potential/measurements;
- average-reference zero sum;
- deterministic repeated solve;
- linearity in the dipole moment;
- scaling with moment amplitude;
- decreasing differences between fixed remote observations under refinement `n=4, 8, 16`.

For refinement, the source is chosen near the center but not on a mesh face, edge, or vertex. A dipole exactly at a grid vertex belongs to several cells, and selecting one local P1 gradient does not define an unambiguous refinement sequence.

Source lookup for repeated solves uses the cached `DOLFINxP1TetraLocator`. This accelerates localization but does not remove the mathematical ambiguity of a source on a shared face, edge, or vertex.

Run:

```bash
TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp RUN_DOLFINX_TESTS=1 \
  pytest test_forward_convergence.py
```
