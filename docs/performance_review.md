# Performance Review

## Executive summary

The project is architecturally ready for profiling: the forward, Green and
inverse layers are separated, and most heavy work has clear module boundaries.
The dominant cost is expected to be Green basis construction because it solves
one Neumann FEM system per measurement channel. For 128 electrodes this means
128 linear solves; for 512 electrodes this means 512 solves.

Recent electrode projection work moved repeated point-in-volume checks behind
cached locator objects. The next performance risks are DOLFINx candidate
location, repeated node-to-dof mapping, Green function memory retention, and
missing transfer-matrix caching in benchmark workflows.

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
| DOLFINx source/candidate location | O(N_points * N_dolfinx_cells) in current helper | High |
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

Observed on this run:

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

This makes transfer construction/candidate DOLFINx location the first measured
bottleneck on the refined mesh for small `max-green-rows`. Green solve scaling
will dominate again as row count grows, but the current transfer-location cost
must be fixed before large candidate grids are practical.

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

## Green solve_all

This is the critical bottleneck. The current algorithm solves `K G_i = M_i^T`
once per measurement row. It is numerically clear and easy to test, but runtime
scales linearly with electrode count.

## DOLFINx candidate location

`locate_candidate_points_in_dolfinx` delegates to
`locate_point_in_dolfinx_p1_tetra_mesh` per point. That helper reconstructs
geometry from `V.dofmap.cell_dofs` and scans cells. For hundreds or thousands of
candidate points this can become expensive. On the `torso_refined.msh` smoke
profile, `build_transfer_matrix` took about 32.9 seconds for only 10 candidates
and 4 retained Green functions, which strongly suggests candidate DOLFINx cell
location dominates the stage.

## Node-to-dof mapping

Green RHS creation calls `create_function_from_meshdata_nodal_values`, which
currently obtains a node-to-dof map through the FEM helper. If this map is not
cached across Green rows, it can become repeated KDTree/coordinate matching
work.

## Central surface projection

Inside checks are now cached through `TetraVolumeLocator`, but central
ray-triangle intersection still loops over surface triangles for each projected
electrode. For 512 electrodes on detailed surface meshes this may be noticeable.

## Benchmark orchestration

Forward and inverse benchmark layers are intentionally simple. They can still
repeat solver construction, Green transfer construction or projection work if
the caller does not cache scenario-level objects.

## Short-term optimization plan

1. Cache node-to-dof maps on `NeumannPoissonSolver` or a shared FEM mapping
   object.
2. Add a reusable DOLFINx P1 tetra locator for candidate/source points.
3. Cache DOLFINx candidate cell ids for source regions and store them in
   transfer metadata.
4. Add `project_only_outside=False` in clipped-sphere examples when points are
   known to be external, after validating the projection convention.
5. Add GreenTransferMatrix cache use to benchmark workflows.
6. Use `--max-green-rows` routinely to estimate Green scaling before full runs.

## Medium-term optimization plan

1. Replace DOLFINx brute-force point location with a reusable locator object or
   DOLFINx geometry/BVH API.
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

## Open questions

- Should Green solves be parallelized by measurement row or batched with a
  block solver?
- Should candidate source regions store DOLFINx cell ids in addition to
  MeshData cell ids after a solver is created?
- Should benchmark scenario metadata include hashes or fingerprints for
  geometry, electrodes and transfer matrices?
- What is the intended upper bound for `num_candidates` and `num_electrodes` in
  production benchmarks?
