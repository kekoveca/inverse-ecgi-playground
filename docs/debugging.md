# Debugging

## ParaView shows weird artifacts

Check the pipeline layer by layer:

1. count and values of nonzero RHS DOFs;
2. source position in DOLFINx cell ordering;
3. MeshData/DOLFINx cell-id correspondence;
4. local DOF ordering;
5. source marker in ParaView;
6. electrode positions and barycentric interpolation;
7. PETSc convergence/nullspace diagnostics.

Do not change the RHS sign until ordering and location errors have been ruled out.

## Check RHS localization

```python
from sources import inspect_point_dipole_rhs_petsc

info = inspect_point_dipole_rhs_petsc(solver, source)
print(info["nonzero_dofs"])
print(info["nonzero_values"])
print(info["local_rhs"])
print(info["local_rhs_sum"])
```

For a scalar P1 tetrahedron, the local RHS is written to four cell DOFs. Four nonzero values are expected for a general moment direction; a special direction may produce an exact zero at one DOF. The local RHS sum must be close to zero.

## Check source location

```python
from sources import inspect_point_dipole_location_petsc

info = inspect_point_dipole_location_petsc(solver, source)

assert info["is_inside_used_dolfinx_cell"]
assert abs(info["barycentric_sum"] - 1.0) < 1e-8
assert info["barycentric_min"] > -1e-8

print(info["declared_position"])
print(info["meshdata_located_cell_id"])
print(info["used_cell_id"])
print(info["dof_cell_center"])
print(info["ordering_warning"])
```

`source.cell_id` belongs to MeshData ordering and is not used by the PETSc assembler by default.

For multiple points, inspect the cached locator directly:

```python
locator = solver.p1_tetra_locator()
cell_ids, barycentric = locator.locate_points(points, return_barycentric=True)
print(locator.metadata)
```

Returned ids are local to the DOLFINx solver, not to `MeshData`.

## Export source marker

```python
from forward import export_dolfinx_function_to_vtx
from sources import create_cell_marker_function

marker = create_cell_marker_function(solver, info["used_cell_id"])
export_dolfinx_function_to_vtx(
    marker,
    "output/source_marker.bp",
    name="source_marker",
)
```

Open `output/source_marker.bp` in ParaView. The marker uses P1 nodal values on the selected cell's DOFs, so its support may also be visible on neighboring cells that share those vertices.

## Check mesh/cell ordering

```python
from sources import compare_meshdata_and_dolfinx_cell_centers

report = compare_meshdata_and_dolfinx_cell_centers(solver, max_cells=1000)
print(report["max_diff"])
print(report["mean_diff"])
print(report["worst_cell_id"])
```

If differences are substantially larger than floating-point tolerance, equal integer ids refer to different cells. Do not use MeshData cell ids as DOLFINx ids.

This difference has been confirmed on torso meshes: source cell ids in MeshData and DOLFINx can differ.

## Export RHS and potential

```python
from forward import export_dolfinx_function_to_vtx, export_forward_result_to_vtx
from sources import assemble_point_dipole_rhs_petsc

rhs = assemble_point_dipole_rhs_petsc(solver, source)
export_dolfinx_function_to_vtx(rhs, "output/rhs.bp", name="rhs")

result = forward.solve(source)
export_forward_result_to_vtx(result, "output/potential.bp")
```

Check `source_marker.bp` first, then `rhs.bp`, and only then interpret `potential.bp`.

## Check electrodes

```python
from geometry import electrode_placement_report, validate_torso_geometry

geometry_report = validate_torso_geometry(torso_geometry)
placement = electrode_placement_report(torso_geometry.electrodes, torso_geometry.volume_mesh)

print(geometry_report.is_valid)
print(geometry_report.errors)
print(placement.mean_distance_to_nearest_node)
print(placement.max_distance_to_nearest_node)
```

Also inspect `MeasurementOperator.electrode_cell_ids` and `electrode_barycentric`. Electrodes must lie inside the selected volume mesh or be projected into a valid location first.

If electrodes lie outside the volume mesh, `build_measurement_operator` centrally projects them onto `surface_mesh` or a boundary extracted from the tetra mesh by default:

```python
op = build_measurement_operator(
    torso_geometry.volume_mesh,
    torso_geometry.electrodes,
    surface_mesh=torso_geometry.surface_mesh,
)

print(op.metadata["electrode_projection"])
```

In a projection report, `surface_cell_ids[i] == -1` is normal for an electrode that was not projected. Inspect `projected_mask`, projection distance, and the tutorial's separate nearest-surface diagnostics together.

`surface_mesh.num_points` may also look unexpectedly large because an extracted triangle mesh can retain the full global point array. The actual number of used surface vertices is:

```python
num_surface_used_vertices = np.unique(surface_mesh.cells.ravel()).size
```

## Checking electrode placement in ParaView

The full inverse example exports `electrodes.bp`, a diagnostic scalar field on
the FEM mesh. It marks the nearest DOLFINx dof to each electrode position:

```python
from forward import export_electrode_markers_to_vtx, inspect_electrode_marker_mapping

info = inspect_electrode_marker_mapping(solver, electrodes)
print(info["num_unique_dofs"])
print(info["num_collisions"])
print(info["max_distance"])

export_electrode_markers_to_vtx(
    solver,
    electrodes,
    "output/electrodes.bp",
    value_mode="index",
)
```

Open `potential.bp`, `source_marker.bp` and `electrodes.bp` together in
ParaView. Marker values `1, 2, 3, ...` correspond to electrode index + 1. This
is not an exact point cloud: if an electrode lies between mesh nodes, the marker
appears at the nearest dof. For exact coordinates and distances, inspect
`electrode_marker_mapping.csv`.

## Solver diagnostics

```python
print(solver.diagnostics.converged_reason)
print(solver.diagnostics.residual_norm)
print(solver.diagnostics.nullspace_test_passed)
```

A positive `converged_reason` and a successful nullspace test are prerequisites for interpreting the physical field shape.

## Green RHS is incompatible

For pure Neumann Green solves every measurement row must sum to zero:

```python
from green import measurement_matrix_row_sums

row_sums = measurement_matrix_row_sums(forward.measurement_operator)
print(np.max(np.abs(row_sums)))
```

`reference="none"` is usually incompatible; use `average` or a correctly configured `single` reference. Do not fix this with arbitrary subtraction after the Green solve: compatibility must be a property of the measurement functional.

## Inverse result has wrong sign or location

First compare the ordinary forward result and Green prediction at the true candidate:

```python
diagnostics = compare_forward_and_green(
    forward_result,
    transfer,
    candidate_index=true_index,
    moment=source.moment,
)
print(diagnostics)
```

If `best_rel_error` is small but inverse selects another position, inspect ambiguity, top candidates, condition numbers, and noise. If the error is large, verify transfer provenance: geometry, electrode order, reference, `measurement_row_indices`, `sigma`, candidates, and `sign`. Inverse uses `transfer.matrix_for_candidate()` and does not change the sign itself.

## If forward solution looks unstable

Run checks from least to most expensive:

```bash
pytest tests/test_convergence_utils.py tests/test_measurements_module.py

TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp RUN_DOLFINX_TESTS=1 \
  pytest tests/test_forward_convergence.py tests/test_poisson_manufactured_solution.py
```

Interpretation:

- manufactured L2 convergence fails: inspect mesh/FEM assembly/nullspace/solver;
- manufactured test passes but linearity or scaling fails: inspect RHS assembly and KSP tolerance;
- linearity passes but refinement measurements are unstable: inspect source location relative to the mesh skeleton and observation ordering;
- all verification tests pass but the torso field looks wrong: inspect physical geometry, conductivity model, source marker, and electrodes.

Do not use a source point exactly on a refinement-grid vertex for cell-local point-dipole convergence: that point belongs to several tetrahedra.

## If transfer matrix construction is slow

Separate point location from gradient evaluation:

```bash
python3 scripts/profile_components.py \
  --component green-transfer \
  --mesh meshes/torso_refined.msh \
  --num-candidates 50 \
  --num-measurements 16 \
  --output output/green_transfer_profile
```

The profile compares a build with lookup against one with prelocated `candidate_cell_ids`. Both paths use the cached `DOLFINxP1TetraLocator`, and basis gradients are computed in batch. If the stage is still expensive, inspect initial locator construction, candidate/function counts, and retained Green-function memory separately.
