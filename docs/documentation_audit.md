# Documentation Audit

## Summary

The documentation was audited against package exports, implementation signatures, runnable examples, profiling scripts and the current test layout. The public narrative now covers the complete forward -> Green -> inverse -> benchmark pipeline instead of presenting inverse work as future functionality.

No mathematical sign, solver, benchmark data-model or production algorithm was changed. One example/script CLI mismatch was fixed: `--reference single` now has a matching `--reference-index` argument passed to the existing measurement API.

## Files reviewed

- root `README.md`;
- all Markdown and JSON reports in `docs/`;
- `examples/`, `scripts/` and their CLI parsers;
- package `__init__.py` exports for geometry, fem, sources, measurements, forward, green, inverse, benchmark, verification and performance;
- module README files;
- test files and DOLFINx gating convention.

## Major updates

- Rebuilt the root README as the main entry point with requirements, forward/inverse commands, outputs, testing and limitations.
- Added [conventions.md](conventions.md) as the canonical ordering/sign/reference/units contract.
- Added [api_overview.md](api_overview.md) from actual `__all__` exports.
- Added [performance.md](performance.md) and [scripts/README.md](../scripts/README.md) for profiling workflows.
- Updated architecture to include Green, inverse, benchmark, verification and performance layers.
- Expanded full-example documentation with physical-group requirements, outputs and troubleshooting.
- Updated architecture/performance audit reports after cached node-to-DOF mapping and `DOLFINxP1TetraLocator` were implemented.

## Stale content removed

- Removed the obsolete `main.py` command in favor of `examples/forward_pipeline.py`.
- Replaced statements that DOLFINx point location scans all cells on every call with the current cached KD-tree/barycentric locator behavior.
- Replaced “sign will be checked later” wording with the current tested `+1` FEM/Green convention.
- Replaced “electrode projection is future work” wording with the current central projection API and diagnostics.
- Removed fixed mesh cell-count expectations from examples/docs.

Historical audit text that quotes old wording remains only where it explicitly documents a completed fix.

## API mismatches fixed

- Full inverse example and full profiling CLI exposed `reference="single"` without a way to supply `reference_index`. Added `--reference-index` and passed it to `build_measurement_operator`.
- Documentation now states that `TaggedMesh` is not exported; `MeshData` owns multi-block/physical-tag behavior.
- `GreenTransferMatrix.measurement_row_indices`, signed `matrix_for_candidate`, cached FEM mapping/locator and electrode marker exports are documented.
- No missing package export was found during `__all__` review.

## Remaining documentation gaps

- There is no pinned dependency/environment manifest, so installation documentation can only name required packages and defer DOLFINx version compatibility to the runtime distribution.
- Markdown examples are syntax-reviewed but are not an automated doctest suite.
- Transfer cache provenance fields are recommended but not enforced by a schema.
- MPI/global ownership behavior remains a documented limitation rather than a supported workflow.
- Exact physical units for a study remain user metadata inherited from the mesh.

## Commands checked

The following parsers were invoked with `--help`:

```text
examples/forward_pipeline.py
examples/full_inverse_experiment_torso.py
examples/full_inverse_experiment_torso_clipped_sphere_electrodes.py
scripts/profile_full_inverse_experiment.py
scripts/profile_components.py
```

Documented test commands are:

```bash
pytest
TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp RUN_DOLFINX_TESTS=1 pytest
```

Runtime experiment commands require the named `.msh` file and a working DOLFINx/ADIOS2 environment.

## Validation results

- `python3 -m compileall .`: passed;
- default `pytest`: 110 passed, 37 DOLFINx-gated tests skipped;
- `TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp RUN_DOLFINX_TESTS=1 pytest`: 147 passed;
- all five documented CLI parsers returned success for `--help`.

## Links checked

Relative Markdown links were checked against the workspace filesystem. JSON audit files were parsed, and fenced code blocks were checked for balance.

## Recommendations

1. Add a pinned container/environment definition for reproducible DOLFINx installation.
2. Add a lightweight documentation CI check for links, JSON and CLI `--help` commands.
3. Promote geometry/electrode/reference/units fingerprints into a required transfer-cache provenance schema.
4. Add doctest-style smoke coverage for short numpy-only examples if documentation grows further.
