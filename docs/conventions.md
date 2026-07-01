# Project conventions

This file is the canonical reference for ordering, sign, reference and coordinate conventions shared across modules.

## Mesh and ordering conventions

### MeshData nodes and cells

`MeshData.points[node_id]` and `MeshData.cells[cell_id]` use the ordering of the geometry container. Numpy source assembly, interpolation matrices and `SourceRegion` use these ids.

- `SourceRegion.candidate_cell_ids`: MeshData cell ids.
- `MeasurementOperator.raw_matrix()` and `.matrix()`: columns in MeshData node ordering.
- `assemble_point_dipole_rhs_numpy`: vector in MeshData node ordering.

### DOLFINx dofs and cells

DOLFINx creates its own function-space and local-cell orderings:

- `function.x.array[dof_id]`: local DOLFINx DOF ordering;
- `V.dofmap.cell_dofs(dolfinx_cell_id)`: local dofs of one DOLFINx cell;
- `GreenTransferMatrix.candidate_cell_ids`: owned local DOLFINx cell ids;
- explicit `cell_id=` in `assemble_point_dipole_rhs_petsc`: local DOLFINx cell id.

MeshData integer ids and DOLFINx integer ids may differ even for the same coordinates.

> Never copy a MeshData-ordered nodal vector directly into a DOLFINx Function or PETSc Vec unless a verified node-to-DOF map is applied.

`NeumannPoissonSolver.p1_node_dof_mapping()` caches a serial scalar-P1 coordinate mapping. `p1_tetra_locator()` caches local DOLFINx cell geometry and returns local DOLFINx ids after barycentric containment checks.

### Source cell semantics

`PointDipole.cell_id` belongs to MeshData ordering. The PETSc assembler does not trust it by default. It locates `source.position` in DOLFINx ordering unless:

- explicit `cell_id=` is passed, or
- `trust_source_cell_id=True` is deliberately enabled after ordering verification.

Candidate points for Green transfer are similarly located from coordinates unless already verified DOLFINx cell ids are supplied.

## Sign conventions

For a point dipole at `x0` with moment `p`, the implemented weak-form RHS is:

```text
integral sigma grad(u) . grad(v) dx = p . grad(v)(x0)
```

For scalar P1 tetra elements:

```python
local_rhs = gradients_p1_tetra(vertices) @ source.moment
```

Green functions use:

```text
K G_i = M_i^T
A[j, i, :] = grad G_i(x_j)
g = A_j p
```

The discrete FEM/Green consistency tests currently confirm sign `+1`. The sign remains explicit in `GreenTransferMatrix.sign`.

Inverse code must use:

```python
A_j = transfer.matrix_for_candidate(j)
```

and not raw `transfer.A[j]`, because the public method applies `transfer.sign`.

## Reference conventions

Raw interpolation and referenced measurements are:

```text
y_raw = P u
g = R y_raw = R P u
M = R P
```

Supported references:

- `none`: no change;
- `average`: subtract the electrode mean;
- `single`: subtract one electrode selected by `reference_index`.

`average` is the default for forward/Green/inverse workflows. It removes the arbitrary constant potential and makes every row of `M` sum to zero. A valid `single` reference also produces zero-sum measurement rows.

`reference="none"` is valid for forward values but is generally incompatible with pure-Neumann Green solves because rows of `M` need not be orthogonal to constants. Use `measurement_matrix_row_sums` or `check_measurement_matrix_compatibility` before Green solves.

## Pure Neumann convention

The stiffness matrix has a constant nullspace:

```text
K 1 = 0
```

`NeumannPoissonSolver` attaches a PETSc constant `NullSpace`, removes the nullspace component from compatible RHS vectors and fixes solution gauge by mean removal. Absolute potential is not physically unique; referenced electrode differences are gauge-invariant.

## Coordinate and unit conventions

The project inherits coordinates and units from the mesh. It does not automatically convert millimeters to meters or register coordinate frames.

The following must share one coordinate frame and unit system:

- `MeshData.points`;
- `ElectrodeSet.positions`;
- `SourceRegion.candidate_points`;
- `PointDipole.position`;
- projection centers and distance tolerances.

Projection distances and localization errors are reported in mesh coordinate units. Conductivity and dipole moment units must be chosen consistently with the intended physical model.

## Surface mesh convention

An extracted triangle `MeshData` may retain the full global point array. Therefore `surface_mesh.num_points` can equal `volume_mesh.num_points` even though boundary triangles reference fewer vertices:

```python
used_surface_vertex_ids = np.unique(surface_mesh.cells.ravel())
```

`ElectrodeProjectionReport.surface_cell_ids[i] == -1` means no projection triangle was recorded for electrode `i`, usually because `project_only_outside=True` left an inside/on-surface electrode unchanged. Inspect `projected_mask` and nearest-surface diagnostics together.

## ParaView marker convention

`electrodes.bp` is a scalar P1 marker at the nearest FEM DOF. Values `1, 2, ...` identify electrode index + 1. It is not an exact electrode point cloud. Exact coordinates and nearest-DOF distances are written to `electrode_marker_mapping.csv` by the full example.
