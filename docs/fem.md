# FEM

`fem` is the DOLFINx/PETSc-backed core for scalar Poisson problems with pure Neumann boundary conditions.

## NeumannPoissonSolver

`NeumannPoissonSolver` is the public alias of `FEMProblem`. It assembles the system for

```text
-div(sigma grad(u)) = f
```

with constant scalar conductivity `sigma` and a homogeneous Neumann boundary condition. The MVP supports P1 Lagrange elements only (`degree=1`).

The solver creates and stores:

- DOLFINx mesh (`domain`);
- function space (`V`);
- stiffness matrix (`K`, alias `A`);
- PETSc constant nullspace;
- a KSP solver reused for multiple RHS vectors;
- a cached scalar-P1 node-to-DOF mapping;
- a cached local-cell tetra locator for source/candidate lookup.

## Pure Neumann nullspace

For a pure Neumann problem,

```text
K 1 = 0
```

so the solution is defined only up to a constant. This is a mathematical property, not an assembly error.

`NeumannNullspaceHandler`:

1. creates a PETSc `NullSpace(constant=True)`;
2. attaches it to `K`;
3. removes the constant component from the RHS;
4. checks RHS compatibility;
5. fixes the solution gauge by removing the mean.

A compatible RHS has zero sum in the discrete constant mode. `solve` performs projection and validation by default.

## Mesh conversion

```text
geometry.MeshData -> dolfinx.mesh.Mesh -> P1 FunctionSpace
```

`create_dolfinx_mesh` performs the conversion.

### Critical ordering warning

The following identities are not guaranteed after conversion:

```text
MeshData node_id == DOLFINx dof_id
MeshData cell_id == DOLFINx cell_id
```

Do not write values indexed by `MeshData` directly into PETSc/FEniCSx vectors. For cell-local operations, use `V.dofmap.cell_dofs(dolfinx_cell_id)` and `V.tabulate_dof_coordinates()`.

### Node-to-DOF mapping

A coordinate-verified mapping is available for serial scalar P1 spaces:

```python
mapping = solver.p1_node_dof_mapping()
node_to_dof = mapping.node_to_dof
dof_to_node = mapping.dof_to_node
```

`build_node_to_dof_map_p1(solver)` delegates to this cache. `forward` and `green` use the mapping when a numpy operator in MeshData node ordering must be connected to a DOLFINx Function. For distributed MPI layouts, the helper deliberately raises instead of guessing ownership or ghost ids.

### P1 tetra locator

```python
locator = solver.p1_tetra_locator()
cell_ids, barycentric = locator.locate_points(points, return_barycentric=True)
cell_dofs, vertices = locator.cell_geometry(cell_ids)
grads_phi = locator.basis_gradients(cell_ids)
```

`DOLFINxP1TetraLocator` caches DOF coordinates, local cell DOFs, vertices, centers, and a KD-tree once. The KD-tree only orders candidate cells; a barycentric containment test always verifies the final match. Returned ids refer to owned local DOLFINx cells. Only scalar P1 tetra spaces are supported.

## Solver usage

```python
from fem import NeumannPoissonSolver

solver = NeumannPoissonSolver(
    mesh=volume_mesh,
    degree=1,
    sigma=1.0,
    ksp_type="cg",
    pc_type="hypre",
)

try:
    rhs = solver.zero_function()
    potential = solver.solve(rhs)
finally:
    solver.destroy()
```

`rhs` may be a `dolfinx.fem.Function`, PETSc `Vec`, or supported vector-like object. The example solves the trivial zero-RHS problem; use `sources.assemble_point_dipole_rhs_petsc` for a point dipole.

Matrix `K` and the KSP setup are not rebuilt between `solve` calls.

## Diagnostics

`solver.diagnostics` is available after assembly and solving:

```python
print(solver.diagnostics.ksp_type)
print(solver.diagnostics.pc_type)
print(solver.diagnostics.converged_reason)
print(solver.diagnostics.residual_norm)
print(solver.diagnostics.nullspace_test_passed)
```

A positive `converged_reason` indicates successful PETSc KSP convergence. `nullspace_test_passed` records the constant-nullspace check performed during solver creation.

## Resource lifetime

PETSc objects are released explicitly:

```python
solver.destroy()
```

`destroy()` also clears FEM mapping/locator caches. Do not reuse locator or mapping objects after destroying the solver.

In an MPI program, creation, solve, and destruction must be performed consistently by all ranks in the communicator.

## Manufactured solution convergence

The smooth solver check uses the unit cube and

```text
u_exact = cos(2 pi x) cos(2 pi y) cos(2 pi z)
-Delta u_exact = 12 pi^2 u_exact
```

The normal derivative is zero on every unit-cube face, so the function is compatible with the homogeneous Neumann boundary condition.

`test_poisson_manufactured_solution.py` solves refinement levels `n=4, 8, 16`, aligns the constant gauge, and checks:

- monotonically decreasing L2 error;
- minimum observed rate greater than `1.0`.

For P1 on a smooth solution, the L2 rate should be close to quadratic, but the test uses a soft threshold to avoid dependence on coarse-grid and solver details.
