# Examples

## Example 1: read `.msh` and build geometry

```python
from geometry import read_gmsh_meshio

tagged = read_gmsh_meshio("torso.msh", dim=3)
volume_mesh = tagged.to_mesh_data("tetra", physical_name="domain")
surface_mesh = tagged.to_mesh_data("triangle", physical_name="boundary")

print(volume_mesh.num_points, volume_mesh.num_cells)
print(tagged.field_data)
```

## Example 2: create source region from bounding box

```python
from geometry import SourceRegion

source_region = SourceRegion.from_bounding_box(
    mesh=volume_mesh,
    bounds_min=[-20.0, -20.0, -20.0],
    bounds_max=[20.0, 20.0, 20.0],
    mode="center",
)

print(source_region.num_candidates)
```

## Example 3: solve forward problem

```python
from fem import NeumannPoissonSolver
from forward import ForwardSolver
from sources import PointDipole

solver = NeumannPoissonSolver(volume_mesh, degree=1, sigma=1.0)
try:
    pipeline = ForwardSolver(poisson_solver=solver)
    source = PointDipole(
        position=[0.0, 0.0, 0.0],
        moment=[0.0, 0.0, 1.0],
    )
    result = pipeline.solve(source)
    print(result.num_nodes)
    print(solver.diagnostics.residual_norm)
finally:
    solver.destroy()
```

## Example 4: export potential and source marker

```python
from fem import NeumannPoissonSolver
from forward import export_dolfinx_function_to_vtx, export_forward_result_to_vtx
from sources import PointDipole, create_cell_marker_function, inspect_point_dipole_location_petsc

solver = NeumannPoissonSolver(volume_mesh, degree=1, sigma=1.0)
try:
    source = PointDipole(position=[0.0, 0.0, 0.0], moment=[0.0, 0.0, 1.0])
    pipeline = ForwardSolver(solver)
    result = pipeline.solve(source)
    info = inspect_point_dipole_location_petsc(solver, source)
    marker = create_cell_marker_function(solver, info["used_cell_id"])

    export_forward_result_to_vtx(result, "output/potential.bp")
    export_dolfinx_function_to_vtx(marker, "output/source_marker.bp", name="source_marker")
finally:
    solver.destroy()
```

Откройте `.bp` outputs в ParaView.

## Example 5: debug source position

```python
from fem import NeumannPoissonSolver
from sources import (
    PointDipole,
    compare_meshdata_and_dolfinx_cell_centers,
    inspect_point_dipole_location_petsc,
)

solver = NeumannPoissonSolver(volume_mesh, degree=1, sigma=1.0)
try:
    source = PointDipole(position=[0.0, 0.0, 0.0], moment=[0.0, 0.0, 1.0])
    info = inspect_point_dipole_location_petsc(solver, source)
    assert info["is_inside_used_dolfinx_cell"]
    assert abs(info["barycentric_sum"] - 1.0) < 1e-8

    print("MeshData cell:", info["meshdata_located_cell_id"])
    print("DOLFINx cell:", info["used_cell_id"])
    print("barycentric:", info["barycentric_in_dolfinx_cell"])

    ordering = compare_meshdata_and_dolfinx_cell_centers(solver, max_cells=1000)
    print("max center difference:", ordering["max_diff"])
finally:
    solver.destroy()
```

## Run the project script

```bash
python3 main.py \
  --mesh torso.msh \
  --physical-name domain \
  --position 0 0 0 \
  --moment 0 0 1
```

Скрипт создаёт `potential.bp`, `potential.xdmf`, `rhs.bp`, `source_marker.bp` и JSON summary в `output/`.
