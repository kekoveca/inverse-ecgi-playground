# Architecture Review

## Executive summary

The project has reached a coherent layered architecture for a full single-dipole pipeline:

```text
geometry -> fem -> sources -> measurements -> forward -> green -> inverse -> benchmark
```

The strongest parts are the explicit separation between `MeshData` and DOLFINx, the documented `(dim, tag)` `field_data` convention, the safe point-dipole RHS assembly in DOLFINx DOF ordering, and the end-to-end tests that verify forward/Green/inverse consistency. The code is small enough that module boundaries remain understandable.

The main architectural risks are not conceptual; they are scale and semantics risks:

- DOLFINx mapping is now cached on the solver, but it remains serial/MVP and will not scale to MPI without a dedicated ownership-aware mapping layer.
- Some public names still hide ordering semantics (`cell_id`, `candidate_cell_ids`), though `ForwardResult` now records potential ordering explicitly.
- Green transfer matrices now carry measurement row ids, but benchmark/provenance metadata is not yet strong enough to prevent stale-cache mismatches.
- Real torso workflows need stronger provenance, projection-quality review, GreenTransferMatrix cache validation and ambiguity diagnostics.

No large refactor was performed in this pass. The originally high-severity ordering/API issues were addressed with small compatibility-preserving changes; remaining items are medium/roadmap risks.

## Current architecture map

```text
geometry
  MeshData, SourceRegion, ElectrodeSet, TorsoGeometry
  owns mesh/electrode/source-region data without DOLFINx

fem
  converts MeshData -> DOLFINx mesh
  creates P1 FunctionSpace
  assembles K
  owns PETSc nullspace/KSP lifecycle

sources
  computes P1 tetra geometry
  assembles point-dipole RHS in numpy MeshData node ordering
  assembles point-dipole RHS directly in DOLFINx DOF ordering

measurements
  locates electrodes in MeshData tetra mesh
  builds P, R, M = R @ P in MeshData node ordering

forward
  source -> RHS -> Neumann solve -> measurements -> ForwardResult
  maps DOLFINx DOF values to MeshData node order before measurement evaluation

green
  measurement rows M_i -> DOLFINx RHS via node-to-dof map
  solves K G_i = M_i^T
  computes A[j, i, :] = grad G_i(x_j)

inverse
  for each candidate, solves 3-variable regularized LS
  chooses minimum residual candidate

benchmark
  forward benchmark: sources x electrode subsets x noise
  inverse benchmark: ForwardBenchmarkResult + GreenTransferMatrix -> reconstruction metrics
```

The dependency direction is mostly clean. `geometry` does not import DOLFINx/PETSc. `fem` owns DOLFINx. `sources`, `measurements`, `green` and `forward` use FEM objects only through explicit adapter-style functions.

## Module-by-module review

### geometry

Good:

- `MeshData` is a lightweight independent container.
- Multi-block Gmsh/meshio import is merged into `MeshData`, avoiding the old `TaggedMesh` split.
- `field_data` is normalized to internal `name -> (dim, tag)`.
- `read_gmsh_meshio` correctly converts meshio `name -> (tag, dim)` into `(dim, tag)`.
- `SourceRegion.candidate_cell_ids` are MeshData cell ids and `TorsoGeometry` validates their range.
- Visualization imports matplotlib lazily through `_require_matplotlib`, so import-time dependency pressure is low.

Risks:

- `read_gmsh_meshio(dim=2)` defaults to 2D, while torso workflows usually need `dim=3`. Examples usually pass `dim=3`, but the default can surprise users.
- `SourceRegion` stores cell centers by default; for point dipoles, candidates on cell centers are safe, but arbitrary source candidates need facet-ambiguity diagnostics.
- Coordinates have no explicit units convention.
- `load_npz_mesh` uses `allow_pickle=True`, acceptable for local trusted files but should be documented as unsafe for untrusted inputs.

### fem

Good:

- `NeumannPoissonSolver`/`FEMProblem` owns mesh, function space, stiffness matrix, nullspace and KSP setup.
- Pure Neumann nullspace handling is explicit: attach nullspace, remove constant RHS component, fix gauge.
- `rhs_from_local_array` warns in its docstring that production assemblers should fill DOF vectors directly.
- `build_node_to_dof_map_p1` is in `fem`, which is the right conceptual home.

Risks:

- `build_node_to_dof_map_p1` is serial only and coordinate-based. It explicitly rejects MPI, which is safe, but any future distributed benchmark will need a real ownership-aware mapping.
- `FEMProblem`/`NeumannPoissonSolver` now owns a cached serial `DOLFINxP1Mapping` with both `node_to_dof` and `dof_to_node`.
- `destroy()` is manual. Benchmarks use `try/finally`, but user code can leak PETSc resources.
- Solver tolerances are mostly PETSc defaults unless tests manually tighten them. This is fine for production flexibility, but benchmark reproducibility should record KSP options.

### sources

Good:

- Numpy RHS and PETSc RHS are separated.
- Numpy `cell_id`/`source.cell_id` semantics are MeshData cell ids.
- PETSc RHS defaults to locating `source.position` in DOLFINx ordering and ignores `source.cell_id` unless `trust_source_cell_id=True`.
- Local RHS is assembled only on `V.dofmap.cell_dofs(cell_id)`.
- Diagnostics include declared position/cell id, MeshData located cell, DOLFINx used cell, dof coordinates, barycentric coordinates, local RHS and nonzero dofs.
- Sign convention is consistently `local_rhs = grads @ source.moment`.

Risks:

- DOLFINx cell location is a full scan over local cells.
- Sources on mesh facets are ambiguous; the first containing cell wins.
- Diagnostics now flag face/edge/vertex barycentric ambiguity, but source-region generation does not yet avoid facets automatically.
- Some public parameters are named `cell_id` even when their semantic order differs by function.

### measurements

Good:

- `MeasurementOperator` clearly states that evaluation uses MeshData node ordering.
- `P`, `R` and `M = R @ P` are separated.
- Average and single reference implementations are simple and test-covered.
- Constant-potential invariance under average reference is tested.
- Electrode location reuses source tetra geometry instead of duplicating barycentric math.

Risks:

- `reference="none"` is valid for forward measurements but incompatible with pure-Neumann Green RHS rows.
- Real electrodes may lie slightly outside a volume mesh; central projection to a surface is available, but real workflows may still need projection-distance QC and alternative projection modes.
- `MeasurementOperator.matrix()` recomputes `R @ P` each call. With scipy sparse this is tolerable for now, but large repeated Green setup should cache `M`.

### forward

Good:

- `ForwardSolver` composes existing modules rather than reimplementing assembly/measurement logic.
- It now maps DOLFINx DOF values to MeshData node ordering before measurement evaluation.
- Export functions are isolated and validate DOLFINx Function shape.
- Generic `export_dolfinx_function_to_vtx` exists for RHS/source marker diagnostics.

Risks:

- `ForwardResult.nodal_values` remains the backward-compatible field name, but `nodal_value_ordering`, `dof_values` and optional `meshdata_nodal_values` now make the ordering explicit.
- `ForwardSolver.solve_potential` does not expose `trust_source_cell_id`; that is good for safety but advanced users may need an explicit diagnostic path.
- Timing metadata is useful but not enough for reproducible solver settings.

### verification

Good:

- Verification utilities are separated from production modules.
- Manufactured Neumann cosine solution avoids point-dipole singularity.
- Forward convergence tests compare referenced measurements rather than raw singular potentials.
- Convergence reports are small and useful.

Risks:

- Manufactured convergence currently supports serial only.
- The manufactured test goes up to `n=128`; it is gated by `RUN_DOLFINX_TESTS`, which is appropriate, but it can be heavy for frequent local runs.

### benchmark

Good:

- Forward and inverse benchmark objects are separated.
- Scenario configs avoid serializing large meshes/arrays.
- Measurements arrays are stored in NPZ with one key per record, avoiding unsafe object arrays.
- Inverse benchmark validates measurement length against `transfer.num_measurements` and rejects mixed electrode subsets for one transfer.
- Noise models are reproducible and tested.

Risks:

- There is not yet a group-by orchestration layer for many electrode subsets and matching GreenTransferMatrices.
- Transfer provenance is not strong enough to guarantee that a cached `A` belongs to the same geometry/electrode/reference/source-region combination as a forward result.
- Solver/KSP settings are not captured in benchmark config.

### green

Good:

- Green RHS is created from `MeasurementOperator.matrix()` through node-to-DOF mapping; no direct MeshData row copy into DOLFINx vectors.
- Measurement matrix compatibility is checked before `solve_all`.
- Transfer tensor convention is clear: `A.shape == (num_candidates, num_measurements, 3)`.
- `transfer.matrix_for_candidate(j)` applies sign.
- Diagnostics compare both signs; integration test confirms current `+1` convention.
- Cache avoids pickle.

Risks:

- `GreenTransferMatrix` now carries `measurement_row_indices`, so partial Green bases are explicit at the transfer level.
- Candidate cell ids in `GreenTransferMatrix` are DOLFINx ids, while `SourceRegion.candidate_cell_ids` are MeshData ids. This is documented, but the shared field name remains risky.
- `keep_functions=False` produces a basis that cannot build a transfer matrix; this is clear but limits memory-saving workflows.
- Metadata in cached transfer matrices is optional and too weak for benchmark-scale reproducibility.

### inverse

Good:

- Inverse uses only `transfer.matrix_for_candidate`, so sign handling stays in GreenTransferMatrix.
- Tikhonov solve handles `lambda_reg=0` with `np.linalg.lstsq` and `lambda_reg>0` with regularized normal equations.
- Result summaries omit large arrays.
- Metrics handle zero moments by returning `NaN` angle.
- Synthetic and DOLFINx end-to-end tests cover recovery.

Risks:

- Equal residual ties choose the first candidate without an ambiguity warning.
- Rank-deficient candidates return a least-squares solution but no warning unless users inspect `condition_number`.
- `candidate_indices` subset results report residual/moment maps in solved-candidate order, not global candidate-index order; this is documented only implicitly.

## Cross-module conventions

### Ordering conventions

Current state:

| Concept | Owner | Meaning |
| --- | --- | --- |
| `MeshData` node id | `geometry` | Row index in `MeshData.points` |
| `MeshData` cell id | `geometry` | Row index in `MeshData.cells` |
| DOLFINx dof id | `fem` | Index in `Function.x.array` |
| DOLFINx cell id | `fem/sources/green` | Local DOLFINx topology cell index |
| `SourceRegion.candidate_cell_ids` | `geometry` | MeshData cell ids |
| `GreenTransferMatrix.candidate_cell_ids` | `green` | DOLFINx cell ids |
| `MeasurementOperator.matrix()` | `measurements` | MeshData node ordering |
| `ForwardResult.nodal_values` | `forward` | DOLFINx DOF ordering, explicitly marked by `nodal_value_ordering` |

The conventions are mostly safe in code. The remaining risk is naming: `cell_id` is used in multiple public APIs with different orderings. Future API cleanup should introduce explicit names while preserving compatibility aliases.

### Sign conventions

Current convention:

```text
source RHS:       local_rhs = gradients_p1_tetra(vertices) @ moment
Green RHS:        K G_i = M_i^T
transfer:         A[j, i, :] = grad G_i(x_j)
prediction:       g = A_j @ p
```

The sign is fixed in `GreenTransferMatrix.sign` and applied by `matrix_for_candidate`. Inverse never changes sign. Tests confirm `best_sign == +1` for the current discrete setup.

### Reference conventions

`average` is the default reference and is compatible with pure Neumann Green RHS. `single` also produces zero-sum rows for P1 interpolation. `none` is useful for raw forward values but generally incompatible with Green solves because each interpolation row sums to 1.

### Units and coordinates

The project assumes mesh, electrode and source coordinates use the same coordinate system. Units are not currently encoded or validated. This should be made explicit in docs and benchmark metadata.

## Numerical correctness risks

1. Facet/vertex source ambiguity can change local RHS/gradient values; diagnostics now flag these cases.
2. Wrong coordinate units or registration offsets can make inverse results look wrong while all numerics are internally consistent.
3. Incompatible reference rows will fail Green solves; this is caught but late.
4. KSP tolerances affect Green/forward consistency at the `1e-5` level unless tightened in tests.
5. Rank-deficient or nearly rank-deficient candidate matrices can make moment estimates unstable even if localization residuals look good.

## API usability review

Forward simulation is reasonably ergonomic:

```python
solver = NeumannPoissonSolver(mesh)
forward = ForwardSolver(solver, electrodes=electrodes)
result = forward.solve(source)
```

Green/inverse is explicit but longer:

```python
green_basis = GreenSolver(solver, forward.measurement_operator).solve_all()
transfer = build_green_transfer_matrix(solver, green_basis, candidate_points)
inverse_result = SingleDipoleInverseSolver(transfer).solve(result.measurements)
```

Benchmark is now usable for both forward and inverse, but full sweeps still require manual orchestration of matching transfer matrices.

Potential future helpers:

- `build_forward_solver_from_geometry(...)`
- `build_green_transfer_from_geometry(...)`
- `run_single_dipole_reconstruction(...)`
- `run_end_to_end_benchmark(...)`
- `group_forward_records_by_measurement_operator(...)`

## Testing review

| Area | Existing tests | Missing tests | Risk |
| --- | --- | --- | --- |
| Geometry import/tags | field_data conversion, MeshData, SourceRegion, TorsoGeometry | real small `.msh` fixture with physical groups | Medium |
| FEM/nullspace | unit mocks, DOLFINx integration, manufactured convergence | MPI/distributed run | Medium |
| Sources | tetra geometry, numpy RHS, PETSc RHS localization, source location diagnostics, facet/edge/vertex flags | source-region filtering that avoids facets | Medium |
| Measurements | interpolation, references, constant invariance, sparse/dense, central projection for outside electrodes | surface-normal/nearest-surface projection alternatives | Low |
| Forward | result/export, DOLFINx solve, convergence, linearity/scaling | larger realistic geometry smoke | Medium |
| Green | compatibility, RHS mapping, gradient, cache, row-indexed transfer matrices, forward/Green consistency | stronger transfer provenance validation | Medium |
| Inverse | LS, sign use, metrics, forward/Green/inverse consistency | ambiguity/tie/rank-deficiency diagnostics | Medium |
| Benchmark | forward records/io, inverse records/io, DOLFINx inverse benchmark smoke | multi-electrode-subset orchestration and provenance validation | Medium |

DOLFINx tests consistently use `RUN_DOLFINX_TESTS=1` and `pytest.importorskip`. Some files use module-level `pytestmark`, others local markers; this is acceptable but can be centralized later.

## Documentation review

Good:

- Top-level README reflects the current full pipeline.
- `docs/architecture.md` covers ordering boundaries.
- `docs/green.md` and `docs/inverse.md` agree on `g = A_j @ p`.
- `docs/benchmark.md` now documents inverse benchmark outputs.
- `docs/debugging.md` covers source location and RHS artifacts.

Gaps:

- Units/coordinate convention should be more prominent.
- Troubleshooting should add explicit cases for Green RHS incompatible reference and inverse mismatch due to wrong transfer provenance.
- Cache/provenance expectations should be documented before real benchmark sweeps.

## Performance and scalability risks

Short-term bottlenecks:

- Serial coordinate-based `node_to_dof` KDTree matching is cached but still not MPI-aware.
- O(num_candidates * num_cells) DOLFINx point location.
- One Green solve per measurement channel.
- Storing all Green functions by default.
- Recomputing `M = R @ P`.

Medium-term bottlenecks:

- No block RHS solve.
- No chunked transfer matrix construction.
- No grouping/orchestration of transfer matrices by electrode set/reference.
- No strong GreenTransferMatrix cache metadata.

Long-term bottlenecks:

- Distributed mesh support is not yet designed.
- Full dense transfer matrices may become too large for high-resolution source regions.
- Inverse currently scans all candidates and all moments eagerly.

## Benchmark readiness

The project is ready for small-to-medium serial single-dipole experiments with synthetic meshes and controlled source regions. It is not yet ready for large production torso benchmarks without adding:

- transfer provenance validation;
- MPI-aware node/dof mapping and cached candidate/cell mappings;
- transfer matrix cache policy;
- projection-quality reports and review thresholds;
- grouped inverse benchmark orchestration;
- ambiguity/rank diagnostics for inverse candidates;
- reproducible solver option metadata.

## Recommended refactor roadmap

### Critical

No immediate critical correctness defect was found in the tested serial P1 single-dipole path.

### High

The initial high-severity issues from this review were addressed in the follow-up pass:

1. Cached FEM mapping object: added serial `DOLFINxP1Mapping` owned by the solver.
2. `ForwardResult.nodal_values` ordering: added explicit ordering metadata and `dof_values`/`meshdata_nodal_values` access.
3. Partial Green row semantics: `GreenTransferMatrix` now carries `measurement_row_indices`.
4. Facet/vertex ambiguity: source diagnostics and Green transfer metadata now flag boundary candidates.

No unresolved high-severity item remains in `architecture_review_issues.json`; the residual work is listed as medium priority.

### Medium

1. Add DOLFINx BVH/batched point location.
2. Add GreenTransferMatrix provenance schema and validation.
3. Add inverse ambiguity metrics: second-best gap, rank, condition thresholds.
4. Add inverse benchmark grouping by electrode set/reference/transfer id.
5. Add projection-distance QC thresholds and optional nearest-surface/surface-normal projection modes.
6. Add context-manager support for solvers.

### Low

1. Normalize DOLFINx test gating helper.
2. Add units/coordinates doc section.
3. Polish naming around `cell_id` in future API.

## Priority issue list

Detailed machine-readable issues are in [`architecture_review_issues.json`](architecture_review_issues.json).

Top issues:

1. `ARCH-001`: cached node-to-DOF mapping remains serial-only.
2. `ARCH-004`: source/candidate facet ambiguity is diagnosed but still a modeling constraint.
3. `ARCH-008`: weak GreenTransferMatrix provenance for benchmarks.
4. `ARCH-006`: DOLFINx point location is still O(num_points * num_cells).
5. `ARCH-009`: inverse benchmark still needs grouped transfer orchestration for many electrode subsets.

## Safe fixes applied in this pass

Initial documentation-level fixes:

- Updated `TorsoGeometry` docstring from "future inverse" to "forward, Green and inverse workflows".
- Updated `docs/green.md` wording from "future inverse solver" to "the inverse solver".

Follow-up high-issue fixes:

- Added `DOLFINxP1Mapping` and cached `FEMProblem.p1_node_dof_mapping()` with `node_to_dof` and `dof_to_node`.
- Added explicit `ForwardResult.nodal_value_ordering`, `ForwardResult.dof_values` and optional `ForwardResult.meshdata_nodal_values`.
- Added `GreenTransferMatrix.measurement_row_indices` and preserved it in transfer cache files.
- Added barycentric boundary classification and propagated source/candidate ambiguity diagnostics.

No mathematical sign convention was changed.

## Open questions

1. Should the project support MPI/distributed DOLFINx in the next milestone, or stay explicitly serial for the first real benchmarks?
2. What coordinate units will real torso meshes use, and should unit metadata be mandatory?
3. Should `GreenTransferMatrix.candidate_cell_ids` remain DOLFINx cell ids, or should it carry both MeshData and DOLFINx ids?
4. How large are expected source regions and electrode sets in the first benchmark paper/experiment?
5. Should inverse benchmark compare clean/noisy observations for multiple `lambda_reg` values in one scenario?
6. Is central projection sufficient for real registered electrodes, or should nearest-surface/surface-normal projection be preferred?
