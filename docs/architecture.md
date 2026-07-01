# Architecture

## Pipeline

```text
geometry
  |
  v
fem
  |
  v
sources + measurements
  |
  v
forward
  |
  v
green
  |
  v
inverse
  |
  v
benchmark
  |
  v
performance / examples / ParaView diagnostics
```

The arrows describe data flow rather than a strict import chain. The forward branch solves `K u = b_source` and evaluates `g = R P u`; the reciprocal branch solves `K G_i = M_i^T` and builds the transfer tensor used by inverse.

Benchmark layer:

```text
geometry / fem / sources / measurements / forward
                         |
                         v
                     benchmark
```

Supporting layers:

```text
verification -> convergence and consistency tests
performance  -> timing, memory and scaling profiles
```

This is a data-flow diagram, not a strict import chain. For example, `sources` contains numpy geometry plus an adapter for an existing FEM solver, while `measurements` can operate entirely without DOLFINx.

## Module responsibilities

### geometry

Stores geometry without a FEniCSx dependency: nodes, connectivity, physical tags, electrodes, source regions, and the `TorsoGeometry` aggregate.

### fem

Converts `MeshData` into a DOLFINx mesh, creates a scalar P1 function space, assembles stiffness matrix `K`, configures the PETSc constant nullspace, and solves systems with multiple RHS vectors. It also owns the cached node-to-DOF mapping and local-cell `DOLFINxP1TetraLocator`.

### sources

Defines `PointDipole`, P1 tetra geometry, and dipole RHS assembly. Numpy assembly uses MeshData node ordering; the FEM adapter writes directly in DOLFINx DOF ordering.

### measurements

Builds interpolation matrix `P` and reference matrix `R`:

```text
y_raw = P u
g = R P u
```

### forward

Composes the existing layers:

```text
source -> rhs -> solve -> nodal values -> measurements -> ForwardResult -> export
```

### green

Converts rows of `M = R @ P` into compatible Neumann RHS vectors, solves Green problems on the same stiffness matrix, and assembles `A[j, i, :] = grad G_i(x_j)`. The transfer matrix predicts measurements as `g = A_j p`; FEM/Green consistency diagnostics verify the sign.

### inverse

Uses an existing `GreenTransferMatrix` and observed measurements. For each candidate it solves a small Tikhonov/LS problem for three moment components and selects the minimum residual. The module does not implement multiple dipoles or alter the sign convention.

### benchmark

Combines geometry, source sets, electrode subsets, and noise models in forward experiments. The inverse benchmark consumes a `ForwardBenchmarkResult` plus a `GreenTransferMatrix` and records localization, moment, and residual metrics.

### verification

Contains a smooth manufactured solution, unit-cube refinement, and convergence-report utilities. It does not participate in production solves.

### performance

Provides timers, memory snapshots, report writers, and profiling CLIs. It measures existing stages without changing mathematical behavior.

## Important separation

- `geometry` does not import FEniCSx and remains suitable for preprocessing and numpy tests.
- `fem` owns DOLFINx/PETSc solver objects and the matrix/KSP lifetime.
- `sources` and `measurements` have numpy-only cores. Their FEM adapters accept already-created DOLFINx objects.
- `forward` does not reassemble the matrix; it uses an existing `NeumannPoissonSolver`. Export imports `dolfinx.io` lazily.
- `green` uses numpy/scipy measurement matrices but creates Green RHS vectors through a verified node-to-DOF mapping.
- `inverse` has no direct DOLFINx dependency; it operates on numpy transfer matrices and measurement vectors.
- `benchmark` does not build a Green transfer matrix inside the inverse runner. It accepts an existing transfer so noise/electrode/lambda sweeps do not rebuild the Green basis unnecessarily.

## Data ownership and ordering boundaries

### Nodes and DOFs

`MeshData.points[node_id]` and `dolfinx.fem.Function.x.array[dof_id]` use different index spaces. Equal integer ids are not guaranteed.

Do not copy a numpy vector assembled by `MeshData node_id` directly into a PETSc/FEniCSx vector without a verified mapping.

### Cells

`MeshData cell_id` and local `DOLFINx cell_id` may also differ. This difference is observable on real torso meshes after DOLFINx mesh creation.

Therefore the PETSc point-dipole RHS locates the cell from `source.position` in DOLFINx ordering by default. The cached `DOLFINxP1TetraLocator` uses a centroid KD-tree for candidates and barycentric containment for verification. An explicit `cell_id` argument to the PETSc adapter means a local DOLFINx cell id.

`SourceRegion.candidate_cell_ids` always stores MeshData cell ids. `GreenTransferMatrix.candidate_cell_ids` stores located local DOLFINx cell ids. These arrays are not interchangeable.

## Physical tags

Internal convention:

```python
field_data: dict[str, tuple[int, int]]
# name -> (dim, tag)
```

This matches Gmsh API ordering. `meshio` returns `(tag, dim)`, and `read_gmsh_meshio` converts each pair during import.

## Pure Neumann problem

The pure-Neumann stiffness matrix has a constant nullspace. Potential is defined only up to an additive constant. `fem` attaches a PETSc `NullSpace`, checks/projects the RHS, and fixes the gauge after solving by removing the mean.

Average-referenced measurements are also invariant to a constant potential shift.

## Units and coordinates

`MeshData.points`, `ElectrodeSet.positions`, `SourceRegion.candidate_points`, and `PointDipole.position` must use one coordinate frame and unit system. `sigma`, localization thresholds, and distance diagnostics must be interpreted consistently; no automatic mm/m conversion is performed.

## Cached spatial data

A single `NeumannPoissonSolver` owns caches associated with its DOLFINx mesh:

- `p1_node_dof_mapping()` — serial scalar-P1 permutation MeshData node ↔ DOLFINx dof;
- `p1_tetra_locator()` - DOF coordinates, local cell DOFs/vertices/centers, and a KD-tree;
- the stiffness matrix and KSP setup for all forward/Green RHS vectors.

Caches are valid only for that solver and are cleared by `solver.destroy()`. The current locator searches owned local cells; distributed global point ownership remains a separate problem.

## Verification layer

`verification` does not participate in the production pipeline. It provides unit-cube meshes, analytic/manufactured functions, and convergence reports for FEM/forward tests. This separation checks smooth FEM convergence independently from the singular point-dipole solution.
