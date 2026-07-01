# Performance profiling

The `performance` package and `scripts/` CLIs measure runtime and memory without changing numerical behavior. For the historical audit and optimization roadmap, see [performance_review.md](performance_review.md).

## Utilities

```python
from performance import (
    PerformanceTimer,
    estimate_array_memory_mb,
    format_timing_table,
    get_process_memory_mb,
    profile_callable,
    run_cprofile,
    save_timing_csv,
    save_timing_json,
)
```

`PerformanceTimer.time(name, **metadata)` is a context manager. Memory sampling uses `psutil` when available and falls back to `resource` on Unix.

## Full pipeline profile

```bash
python3 scripts/profile_full_inverse_experiment.py \
  --mesh torso_refined.msh \
  --output output/performance_profile \
  --num-electrodes 128 \
  --num-candidates 50 \
  --max-green-rows 8 \
  --no-export
```

The mesh must contain `domain` tetra and `boundary` triangle physical groups unless different names are supplied.

Outputs:

```text
timing.csv
timing.json
memory.json
profile_summary.md
```

`timing.csv` is suitable for spreadsheets. `timing.json` preserves stage metadata. `memory.json` records snapshots after mesh, solver, Green and transfer stages. `profile_summary.md` sorts the slowest stages and includes run metadata.

Useful controls:

- `--max-green-rows N`: solve only the first N measurement Green problems;
- `--skip-green`: profile geometry/forward only;
- `--skip-inverse`: build Green transfer without inverse search;
- `--no-export`: remove VTX I/O from timings.

When a partial Green basis is used, transfer measurement rows are identified by `measurement_row_indices`; inverse profiling selects the corresponding observations.

## Component profiles

### Point location

```bash
python3 scripts/profile_components.py \
  --component point-location \
  --mesh torso_refined.msh \
  --num-location-points 100 \
  --output output/point_location_profile
```

This measures initial `DOLFINxP1TetraLocator` construction and batched location separately.

### Green transfer construction

```bash
python3 scripts/profile_components.py \
  --component green-transfer \
  --mesh torso_refined.msh \
  --num-candidates 50 \
  --num-measurements 16 \
  --output output/green_transfer_profile
```

Synthetic DOLFINx functions isolate transfer construction from Green linear solves. The report compares lookup inside `build_green_transfer_matrix` with already located DOLFINx cell ids.

### Numpy inverse scaling

```bash
python3 scripts/profile_components.py \
  --component inverse-scaling \
  --candidate-counts 100 1000 10000 \
  --measurement-counts 32 128 \
  --output output/inverse_scaling_profile
```

This component does not require DOLFINx.

## Current bottlenecks

1. `GreenSolver.solve_all()` performs one linear solve per measurement channel.
2. Retaining one DOLFINx Green Function per channel can dominate memory.
3. A dense transfer tensor costs `num_candidates * num_measurements * 3 * 8` bytes for float64.
4. Central surface projection tests triangles for each projected electrode.
5. Very large inverse candidate sets use a Python loop of small LS solves.

Point location and node-to-DOF mapping are cached for the current serial P1 path. The locator uses KD-tree candidates and barycentric verification; distributed/global MPI ownership is not implemented.

## Reading results

Compare stage times only between runs with recorded mesh/electrode/candidate counts and the same solver settings. A small `--max-green-rows` run estimates per-row solve time but does not capture every cache/preconditioner effect of a full basis.

Run component profiles when `build_transfer_matrix` looks unexpectedly slow. If prelocated and normal builds are similar, gradient extraction or memory traffic dominates; if they differ, inspect locator construction and candidate geometry.
