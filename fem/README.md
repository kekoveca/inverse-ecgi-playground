# FEM module

Full guide: [../docs/fem.md](../docs/fem.md).

`fem` is the DOLFINx/PETSc backend for scalar Poisson problems with pure Neumann boundary conditions. It owns the mesh conversion, P1 space, stiffness matrix, constant nullspace, KSP solver, and solver-scoped ordering caches.

## Main objects

- `FEMProblem` / `NeumannPoissonSolver`
- `FunctionSpaceFactory`
- `StiffnessOperator`
- `NeumannNullspaceHandler`
- `LinearSolver`
- `SolverDiagnostics`
- `DOLFINxP1Mapping`
- `DOLFINxP1TetraLocator`

The current MVP supports scalar P1 elements and constant scalar conductivity.

## Minimal example

```python
from fem import NeumannPoissonSolver

solver = NeumannPoissonSolver(
    volume_mesh,
    degree=1,
    sigma=1.0,
    ksp_type="cg",
    pc_type="hypre",
)
try:
    rhs = solver.zero_function()
    potential = solver.solve(rhs)
    print(solver.diagnostics)
finally:
    solver.destroy()
```

The stiffness matrix and KSP setup are reused across solves.

## Pure Neumann nullspace

The stiffness matrix satisfies `K 1 = 0`. The solver:

1. creates and attaches a constant PETSc `NullSpace`;
2. removes the constant component from the RHS;
3. checks RHS compatibility;
4. fixes the solution gauge by removing its mean.

This nullspace is a property of the mathematical problem, not an assembly error.

## Ordering helpers

```python
mapping = solver.p1_node_dof_mapping()
node_to_dof = mapping.node_to_dof

locator = solver.p1_tetra_locator()
cell_ids = locator.locate_points(points)
gradients = locator.basis_gradients(cell_ids)
```

The mapping and locator are cached per solver. They support the current serial scalar-P1 path and return owned local DOLFINx ids. They are not global MPI ownership maps.

Never copy MeshData-ordered values directly into a DOLFINx Function without applying the verified node-to-DOF map.

## RHS and repeated solves

`solve(rhs)` accepts a DOLFINx Function, compatible vector-like object, or PETSc Vec. Production source assemblers should write directly in DOLFINx DOF ordering. `rhs_from_local_array` is intended for tests and prototypes.

```python
for rhs in rhs_list:
    potential = solver.solve(rhs)
```

## Resource lifetime

Call `solver.destroy()` when finished. It destroys PETSc owners and clears the cached mapping and locator. Do not reuse cache objects after destruction.

## Dependencies and tests

FEniCSx imports are lazy. Real solves require DOLFINx, UFL, Basix, PETSc, `mpi4py`, and `petsc4py`.

```bash
pytest test_fem_units.py
TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp RUN_DOLFINX_TESTS=1 pytest test_fem_dolfinx_integration.py test_fem_p1_locator.py
```

Dipole physics, electrode measurements, Green functions, and inverse regularization belong to higher-level modules.
