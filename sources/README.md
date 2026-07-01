# Sources module

Full guide: [../docs/sources.md](../docs/sources.md).

`sources` implements point-dipole geometry and RHS assembly for scalar P1 tetrahedral FEM.

## Convention

For a dipole at `x0` with moment `p`:

```text
b_i = p . grad(phi_i)(x0)
```

The implemented local array is:

```python
local_rhs = gradients_p1_tetra(vertices) @ source.moment
```

Its sum is zero because the P1 basis gradients sum to zero. The current FEM/Green consistency tests confirm transfer sign `+1`.

## Numpy assembly

```python
from sources import PointDipole, assemble_point_dipole_rhs_numpy

source = PointDipole(
    position=[0.25, 0.25, 0.25],
    moment=[1.0, 0.0, 0.0],
)
rhs = assemble_point_dipole_rhs_numpy(mesh, source)
```

This RHS uses MeshData node ordering. `source.cell_id`, when set, is a MeshData cell id.

## DOLFINx assembly

```python
from sources import assemble_point_dipole_rhs_petsc

rhs = assemble_point_dipole_rhs_petsc(solver, source)
potential = solver.solve(rhs)
```

The adapter locates `source.position` through the cached DOLFINx P1 locator, gets `V.dofmap.cell_dofs(cell_id)`, and writes only the local cell contribution. It never copies the MeshData-ordered numpy RHS into a DOLFINx vector.

By default, `source.cell_id` is not trusted as a DOLFINx id. Explicit `cell_id=` means a local DOLFINx cell id. Use `trust_source_cell_id=True` only after verifying ordering.

## Diagnostics

```python
from sources import inspect_point_dipole_location_petsc, inspect_point_dipole_rhs_petsc

location = inspect_point_dipole_location_petsc(solver, source)
rhs_info = inspect_point_dipole_rhs_petsc(solver, source)

print(location["used_cell_id"])
print(location["barycentric_in_dolfinx_cell"])
print(rhs_info["nonzero_dofs"])
print(rhs_info["local_rhs_sum"])
```

For visual diagnostics:

```python
from forward import export_dolfinx_function_to_vtx
from sources import create_cell_marker_function

marker = create_cell_marker_function(solver, location["used_cell_id"])
export_dolfinx_function_to_vtx(marker, "output/source_marker.bp", name="source_marker")
```

`compare_meshdata_and_dolfinx_cell_centers` helps demonstrate that equal integer cell ids need not identify the same cell.
