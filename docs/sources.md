# Sources

`sources` implements a point-dipole source for scalar P1 FEM on tetrahedral meshes. Geometric calculations are available in numpy, while a separate adapter assembles the RHS in DOLFINx ordering.

## PointDipole

```python
from sources import PointDipole

source = PointDipole(
    position=[0.0, 0.0, 0.0],
    moment=[0.0, 0.0, 1.0],
    cell_id=None,
    name="point_dipole",
    metadata={},
)
```

Fields:

- `position` - source coordinate, shape `(3,)`;
- `moment` - dipole moment, shape `(3,)`;
- `cell_id` - optional MeshData cell id for numpy workflows;
- `name`, `metadata` - user metadata.

`with_cell_id` returns a new immutable object. `normalized_moment` returns the direction of a nonzero moment.

## P1 tetra geometry

The module provides:

- `tetra_signed_volume`, `tetra_volume`;
- `barycentric_coordinates_tetra`;
- `point_in_tetra`;
- `gradients_p1_tetra`.

`gradients_p1_tetra(vertices)` returns `grads[a] = grad(phi_a)` in the order of the four supplied vertices.

## Numpy RHS

```python
from sources import assemble_point_dipole_rhs_numpy

rhs_numpy = assemble_point_dipole_rhs_numpy(volume_mesh, source)
```

This function assembles

```text
b_i = moment . grad(phi_i)
```

in **MeshData node ordering**. It is intended for numpy tests, geometric checks, and workflows without FEniCSx.

If `source.cell_id` is set, the numpy assembler interprets it as a MeshData cell id.

## PETSc/FEniCSx RHS

```python
from sources import assemble_point_dipole_rhs_petsc

rhs = assemble_point_dipole_rhs_petsc(solver, source)
potential = solver.solve(rhs)
```

The PETSc adapter does not copy the numpy RHS. It:

1. locates `source.position` among local DOLFINx cells;
2. obtains `cell_dofs = solver.V.dofmap.cell_dofs(cell_id)`;
3. obtains cell geometry in the same local order from the cached P1 locator;
4. computes `local_rhs = grads @ source.moment`;
5. writes the values directly into a DOLFINx Function.

By default, `source.cell_id` is **not treated as a DOLFINx cell id**. The field belongs to MeshData ordering.

An explicit `cell_id=` in `assemble_point_dipole_rhs_petsc` means a DOLFINx cell id. `trust_source_cell_id=True` allows `source.cell_id` to be used as a DOLFINx id, but only after verifying the ordering.

## Sign convention

Current convention:

```python
local_rhs = gradients_p1_tetra(vertices) @ source.moment
```

Diagnostic helpers deliberately do not alter the sign. The current discrete FEM/Green consistency test confirms `g = A_j @ p` with sign `+1`; the physical orientation and units of a specific model must remain consistent.

## Source location debugging

### DOLFINx point location

```python
from sources import locate_point_in_dolfinx_p1_tetra_mesh

cell_id = locate_point_in_dolfinx_p1_tetra_mesh(solver, source.position)
```

The public wrapper uses the cached `fem.DOLFINxP1TetraLocator` and returns an owned local DOLFINx cell id. The locator queries nearby centroids through a KD-tree and verifies the cell with barycentric coordinates; in the worst case, candidates expand to all local cells.

For batched lookup, use the locator directly:

```python
locator = solver.p1_tetra_locator()
cell_ids = locator.locate_points(candidate_points)
```

### Full diagnostics

```python
from sources import inspect_point_dipole_location_petsc

info = inspect_point_dipole_location_petsc(solver, source)
print(info["used_cell_id"])
print(info["cell_dofs"])
print(info["barycentric_in_dolfinx_cell"])
print(info["is_inside_used_dolfinx_cell"])
print(info["ordering_warning"])
```

`inspect_point_dipole_rhs_petsc` is a backward-compatible name for the same complete diagnostics.

RHS check:

```python
info = inspect_point_dipole_rhs_petsc(solver, source)
print(info["nonzero_dofs"])
print(info["local_rhs"])
print(info["local_rhs_sum"])
```

For a general moment on a P1 tetrahedron, four cell DOFs are expected; individual values may be exactly zero for special moment directions.

### Compare cell ids geometrically

```python
from sources import compare_meshdata_and_dolfinx_cell_centers

report = compare_meshdata_and_dolfinx_cell_centers(solver, max_cells=1000)
print(report["max_diff"])
print(report["worst_cell_id"])
```

Large differences show that equal integer ids identify different cells.

### Source marker

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

Open `source_marker.bp` in ParaView to inspect the cell actually used.
