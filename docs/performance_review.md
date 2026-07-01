# Performance Review

## Executive summary

The project is architecturally ready for profiling: the forward, Green and
inverse layers are separated, and most heavy work has clear module boundaries.
The dominant cost is expected to be Green basis construction because it solves
one Neumann FEM system per measurement channel. For 128 electrodes this means
128 linear solves; for 512 electrodes this means 512 solves.

Recent work moved repeated point-in-volume checks, node-to-dof mapping and
DOLFINx source/candidate lookup behind cached locator/mapping objects. Green
transfer construction now batches cell geometry and basis-gradient evaluation.
The main remaining risks are Green solve count, retained Green-function memory,
central surface ray projection and missing provenance-aware transfer caching in
benchmark workflows.

This review adds lightweight profiling utilities and scripts:

```bash
python3 scripts/profile_full_inverse_experiment.py \
  --mesh torso_refined.msh \
  --output output/performance_profile \
  --num-electrodes 128 \
  --num-candidates 50 \
  --max-green-rows 16 \
  --no-export
```

## Pipeline timing model

```text
read mesh
  -> MeshData extraction
  -> electrode placement/projection
  -> source candidate generation
  -> DOLFINx mesh + FunctionSpace + stiffness matrix
  -> forward RHS + solve + measurements
  -> measurement matrix compatibility
  -> Green solve per measurement row
  -> transfer gradients at candidates
  -> inverse candidate sweep
  -> optional export
```

## Expected complexity by module

| Stage | Current likely complexity | Risk |
| --- | ---: | --- |
| Gmsh read and MeshData extraction | O(points + cells) | Medium |
| Electrode projection inside checks | O(N_elec log N_cells + verified candidates) | Medium |
| Central ray projection | O(N_projected * N_surface_cells) | High |
| SourceRegion bounding box | O(N_cells) | Medium |
| MeshData point location | KDTree over centroids plus barycentric checks | Medium |
| DOLFINx source/candidate location | KD-tree candidates + barycentric verification; worst-case O(N_points * N_local_cells) | Medium |
| Stiffness assembly | FEM assembly over all cells | High |
| One forward solve | solve(K) | High |
| Green solve_all | O(N_meas * solve(K)) | Critical |
| Transfer gradients | O(N_meas * N_candidates) | Medium |
| Single-dipole inverse | O(N_candidates * N_meas) small 3x3 LS per candidate | Low/Medium |
| VTX export | O(num_dofs) per function | Low/Medium |

## Measured timings

The profiling scripts produce run-specific measurements in:

```text
output/performance_profile/
  timing.csv
  timing.json
  memory.json
  profile_summary.md
```

Use `--max-green-rows` for early estimates. For example, if 8 Green rows take
40 seconds, the rough 128-row solve estimate is about 640 seconds before
parallelization or better preconditioner reuse.

Smoke run on `torso_refined.msh`:

```bash
TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp \
python3 scripts/profile_full_inverse_experiment.py \
  --mesh torso_refined.msh \
  --output output/performance_profile_smoke \
  --num-electrodes 16 \
  --num-candidates 10 \
  --max-green-rows 4 \
  --no-export
```

Observed on this historical run, before `DOLFINxP1TetraLocator` was added:

| Stage | Time, s |
| --- | ---: |
| read_mesh | 4.429 |
| create_solver | 0.877 |
| forward_rhs | 3.876 |
| forward_solve | 0.697 |
| green_solve_all, 4 rows | 1.130 |
| green_solve_per_row_mean | 0.282 |
| build_transfer_matrix, 10 candidates x 4 rows | 32.898 |
| inverse_solve | 0.0017 |
| total | 44.384 |

This run identified transfer construction/candidate location as the first
measured bottleneck and motivated the locator fix. Do not treat the 32.898 s
number as current performance: rerun the commands below on the target mesh.

Post-fix synthetic component smoke profile (small unit cube, 5 candidates and
4 synthetic functions) measured approximately 0.8 ms with lookup and 0.3 ms
with prelocated cells. These values validate removal of repeated full scans but
are not a replacement for a torso-mesh profile.

Component-only inverse scaling can be measured without DOLFINx:

```bash
python3 scripts/profile_components.py \
  --component inverse-scaling \
  --output output/inverse_scaling_profile \
  --candidate-counts 100 1000 10000 \
  --measurement-counts 32 128
```

## Memory usage

The main memory growth points are:

- DOLFINx mesh and stiffness matrix;
- retained Green functions when `GreenSolver(keep_functions=True)`;
- dense `GreenTransferMatrix.A` with shape `(num_candidates, num_measurements, 3)`;
- any dense measurement matrix fallback when scipy sparse is unavailable.

Approximate transfer matrix storage:

```text
num_candidates * num_measurements * 3 * 8 bytes
```

For 100k candidates and 512 measurements, this is about 1.23 GB just for `A`.
Retained Green functions can be larger because each function stores one value
per FEM dof.

## Bottlenecks

### Green solve_all

This is the critical bottleneck. The current algorithm solves `K G_i = M_i^T`
once per measurement row. It is numerically clear and easy to test, but runtime
scales linearly with electrode count.

### DOLFINx candidate location

Resolved for the current serial/local P1 path: `DOLFINxP1TetraLocator` caches
dof coordinates, local cell dofs/vertices/centers and a centroid KD-tree.
`locate_candidate_points_in_dolfinx` performs a batched lookup and verifies
hits with barycentric coordinates. Source RHS assembly reuses the same locator.

Residual risks remain: only owned local cells are searched, highly nonuniform
meshes may force candidate-set expansion, and distributed/global ownership is
not implemented.

### Node-to-dof mapping

Resolved for the current serial P1 path: Green RHS creation delegates to the
cached `FEMProblem.p1_node_dof_mapping()`. Coordinate matching is performed once
per solver/tolerance and shared by forward and Green adapters.

### Central surface projection

Inside checks are now cached through `TetraVolumeLocator`, but central
ray-triangle intersection still loops over surface triangles for each projected
electrode. For 512 electrodes on detailed surface meshes this may be noticeable.

### Benchmark orchestration

Forward and inverse benchmark layers are intentionally simple. They can still
repeat solver construction, Green transfer construction or projection work if
the caller does not cache scenario-level objects.

## Short-term optimization plan

1. Record geometry/electrode/reference/conductivity provenance with cached
   Green transfer matrices.
2. Cache DOLFINx candidate cell ids between repeated transfer builds on the
   same solver/source region.
3. Add `project_only_outside=False` in clipped-sphere examples when points are
   known to be external, after validating the projection convention.
4. Add GreenTransferMatrix cache use to benchmark workflows.
5. Use `--max-green-rows` routinely to estimate Green scaling before full runs.
6. Profile central projection separately when many electrodes are outside.

## Medium-term optimization plan

1. Add distributed/global point ownership or a DOLFINx BVH path when MPI support
   becomes a requirement.
2. Vectorize central ray-triangle projection or add a surface acceleration
   structure.
3. Add chunked transfer matrix construction for large candidate grids.
4. Add optional `keep_functions=False` workflows that compute transfer chunks
   without storing all Green functions.
5. Run Green solves in parallel over measurement rows when the execution model
   allows it.

## Long-term optimization plan

1. Batch RHS solves or use block Krylov/direct factorization when appropriate.
2. Matrix-free inverse search for very large candidate sets.
3. Reduced-order Green basis or electrode subset selection for large benchmarks.
4. Memory-mapped transfer matrices for repeated inverse experiments.

## Benchmark commands

Full pipeline:

```bash
TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp \
python3 scripts/profile_full_inverse_experiment.py \
  --mesh torso_refined.msh \
  --output output/performance_profile \
  --num-electrodes 128 \
  --num-candidates 50 \
  --max-green-rows 16 \
  --no-export
```

Numpy-only inverse scaling:

```bash
python3 scripts/profile_components.py \
  --component inverse-scaling \
  --output output/inverse_scaling_profile
```

Point-location profile:

```bash
python3 scripts/profile_components.py \
  --component point-location \
  --mesh torso_refined.msh \
  --output output/point_location_profile
```

Transfer build without Green solves:

```bash
python3 scripts/profile_components.py \
  --component green-transfer \
  --mesh torso_refined.msh \
  --num-candidates 50 \
  --num-measurements 16 \
  --output output/green_transfer_profile
```

## Open questions

- Should Green solves be parallelized by measurement row or batched with a
  block solver?
- Should benchmark orchestration persist locator-produced DOLFINx cell ids for
  one solver lifetime, while keeping them separate from MeshData source ids?
- Should benchmark scenario metadata include hashes or fingerprints for
  geometry, electrodes and transfer matrices?
- What is the intended upper bound for `num_candidates` and `num_electrodes` in
  production benchmarks?
