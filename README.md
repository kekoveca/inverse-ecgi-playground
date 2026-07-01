# Cardio direct/inverse benchmark

## Project goal

The project solves the forward problem and a discrete single-dipole inverse problem for electric potential in a tetrahedral torso model:

```text
meshes/torso.msh -> geometry -> Neumann FEM -> point-dipole forward measurements
          -> GreenTransferMatrix -> single-dipole inverse -> benchmark metrics
```

Numerical verification, ParaView export, and performance profiling accompany the main pipeline.

## Current capabilities

- Gmsh/meshio mesh and physical-group import;
- `MeshData`, electrodes, source regions, and geometry validation;
- a scalar P1 DOLFINx/PETSc Poisson solver with a pure-Neumann nullspace;
- point-dipole RHS assembly in MeshData node and DOLFINx DOF ordering;
- the interpolation/reference operator `M = R @ P`;
- forward solves, measurements, and VTX/XDMF export;
- central projection of outside electrodes onto a triangle surface;
- a Green basis and tensor `A[j, i, :] = grad G_i(x_j)`;
- single-dipole Tikhonov/LS inverse search;
- forward/inverse benchmark records, noise models, and metrics;
- convergence checks, profiling, and diagnostics.

## Repository structure

| Path | Responsibility |
| --- | --- |
| `geometry/` | FEniCSx-independent mesh, electrode and source-region data |
| `fem/` | DOLFINx mesh, P1 space, stiffness matrix, nullspace, KSP and cached mappings |
| `sources/` | Point dipole geometry, RHS assembly and source diagnostics |
| `measurements/` | Electrode location/projection, interpolation and references |
| `forward/` | Forward orchestration, results and ParaView export |
| `green/` | Green RHS, basis, gradients, transfer tensor and cache |
| `inverse/` | Single-dipole reconstruction and metrics |
| `benchmark/` | Forward/inverse scenarios, runners, noise and result I/O |
| `verification/` | Manufactured solutions, refinement and convergence reports |
| `performance/` | Timing, memory and cProfile helpers |
| `examples/` | Runnable forward and full inverse tutorials |
| `scripts/` | Full-pipeline and component profiling CLIs |
| `tests/` | Numpy and gated DOLFINx test suites |
| `meshes/` | Example Gmsh meshes used by tutorials and integration tests |

## Installation / requirements

The repository currently has no packaged installer or pinned environment file. Run commands from the repository root in an environment containing:

- Python 3.10+;
- `numpy`, `scipy`, `meshio`;
- `pytest` for tests;
- DOLFINx, PETSc, `mpi4py`, `petsc4py`, UFL and Basix for FEM workflows;
- ADIOS2/VTX support for `.bp` export;
- optional `matplotlib` for geometry plots and `psutil` for process-memory sampling.

Numpy-only modules and tests import without DOLFINx. The exact compatible DOLFINx stack is environment-specific, so install it through the distribution/container used for the solver.

## Quick start: forward solve

The included CLI reads the `domain` tetra physical group, solves one source and exports diagnostics:

```bash
python3 examples/forward_pipeline.py \
  --mesh meshes/torso.msh \
  --physical-name domain \
  --position 0 0 0 \
  --moment 0 0 1
```

The source position must lie inside the selected volume mesh. Outputs default to `output/potential.bp`, `output/potential.xdmf`, `output/rhs.bp`, `output/source_marker.bp` and `output/forward_summary.json`.

Minimal Python API:

```python
from fem import NeumannPoissonSolver
from forward import ForwardSolver, export_forward_result_to_vtx
from geometry import ElectrodeSet, read_gmsh_meshio
from sources import PointDipole

tagged = read_gmsh_meshio("meshes/torso.msh", dim=3)
volume_mesh = tagged.to_mesh_data("tetra", physical_name="domain")
electrodes = ElectrodeSet(
    positions=volume_mesh.points[[0, 1]].copy(),
    labels=["E1", "E2"],
)

solver = NeumannPoissonSolver(volume_mesh, degree=1, sigma=1.0)
try:
    pipeline = ForwardSolver(solver, electrodes=electrodes, reference="average")
    source = PointDipole(position=[0.0, 0.0, 0.0], moment=[0.0, 0.0, 1.0])
    result = pipeline.solve(source)
    export_forward_result_to_vtx(result, "output/potential.bp")
finally:
    solver.destroy()
```

## Full inverse experiment on torso.msh

The tutorial requires `domain` (dim 3) and `boundary` (dim 2) physical groups:

```bash
python3 examples/full_inverse_experiment_torso.py \
  --mesh meshes/torso.msh \
  --output output/full_inverse_experiment \
  --num-electrodes 32 \
  --num-candidates 50 \
  --moment 0 0 1 \
  --snr-db 40 \
  --lambda-reg 1e-10
```

It creates:

```text
experiment_summary.json
inverse_summary.json
measurements.npz
electrodes.csv
electrode_surface_diagnostics.csv
candidates.csv
electrode_marker_mapping.csv
potential.bp
rhs.bp
true_source_marker.bp
estimated_source_marker.bp
electrodes.bp
```

The clipped-sphere variant generates quasi-uniform outer points and centrally projects them onto the torso:

```bash
python3 examples/full_inverse_experiment_torso_clipped_sphere_electrodes.py \
  --mesh meshes/torso.msh \
  --output output/full_inverse_experiment_clipped_sphere \
  --num-electrodes 32 \
  --num-candidates 50
```

## ParaView outputs

Open `.bp` outputs directly in ParaView. `potential.bp` is the FEM potential, `rhs.bp` is the dipole RHS, and source marker fields identify the true/estimated DOLFINx cells.

`electrodes.bp` is a diagnostic P1 nodal marker at the nearest FEM DOF, not an exact point cloud. Use `electrode_marker_mapping.csv` for actual electrode coordinates and nearest-DOF distances. XDMF remains supported; open the `.xdmf` file and keep its `.h5` companion beside it. Prefer VTX/BP if XDMF is empty or unstable in ParaView.

## Green functions and inverse reconstruction

For average-referenced measurements:

```text
K G_i = M_i^T
A[j, i, :] = grad G_i(x_j)
g = A_j p
```

```python
from green import GreenSolver, build_green_transfer_matrix
from inverse import SingleDipoleInverseSolver

green_basis = GreenSolver(solver, pipeline.measurement_operator).solve_all()
transfer = build_green_transfer_matrix(solver, green_basis, candidate_points)
inverse_result = SingleDipoleInverseSolver(
    transfer,
    lambda_reg=1e-10,
    reference="average",
).solve(result.measurements)
```

The inverse layer always uses `transfer.matrix_for_candidate(j)`, which applies `GreenTransferMatrix.sign`.

## Benchmarking

`ForwardBenchmarkRunner` generates clean/noisy synthetic records. `InverseBenchmarkRunner` consumes a matching `GreenTransferMatrix`:

```python
from benchmark import run_inverse_benchmark, save_inverse_benchmark_result

inverse_result = run_inverse_benchmark(
    forward_result,
    transfer,
    lambda_reg=1e-10,
)
save_inverse_benchmark_result(inverse_result, "results/inverse")
```

One inverse scenario currently corresponds to one electrode subset/reference/transfer definition.

## Performance profiling

Full-pipeline profile:

```bash
python3 scripts/profile_full_inverse_experiment.py \
  --mesh meshes/torso_refined.msh \
  --output output/performance_profile \
  --num-electrodes 128 \
  --num-candidates 50 \
  --max-green-rows 8 \
  --no-export
```

This writes `timing.csv`, `timing.json`, `memory.json` and `profile_summary.md`. Component profiles isolate point location, transfer construction and inverse scaling:

```bash
python3 scripts/profile_components.py --component point-location --mesh meshes/torso_refined.msh
python3 scripts/profile_components.py --component green-transfer --mesh meshes/torso_refined.msh
python3 scripts/profile_components.py --component inverse-scaling
```

## Testing

Numpy/scipy suite:

```bash
pytest
```

DOLFINx/MPI integration suite:

```bash
TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp RUN_DOLFINX_TESTS=1 pytest
```

Without `RUN_DOLFINX_TESTS=1`, DOLFINx tests are intentionally skipped.

## Documentation map

- [Architecture](docs/architecture.md)
- [Conventions](docs/conventions.md)
- [API overview](docs/api_overview.md)
- [Geometry](docs/geometry.md)
- [FEM](docs/fem.md)
- [Sources](docs/sources.md)
- [Measurements](docs/measurements.md)
- [Forward](docs/forward.md)
- [Green](docs/green.md)
- [Inverse](docs/inverse.md)
- [Benchmark](docs/benchmark.md)
- [Performance](docs/performance.md)
- [Examples](docs/examples.md)
- [Debugging](docs/debugging.md)
- [Documentation audit](docs/documentation_audit.md)

## Important conventions

- MeshData node/cell ids are not DOLFINx DOF/cell ids.
- `SourceRegion.candidate_cell_ids` are MeshData ids; `GreenTransferMatrix.candidate_cell_ids` are local DOLFINx ids.
- Never copy MeshData-ordered nodal values into a DOLFINx Function without the node-to-DOF map.
- Point-dipole RHS uses `local_rhs = gradients_p1_tetra(vertices) @ moment`.
- Measurements use `y_raw = P u`, `g = R P u`; `average` is the default reference.
- The pure-Neumann potential is defined up to a constant; PETSc nullspace handling and referenced measurements remove that ambiguity.
- Mesh, electrode and source coordinates must share one frame and unit system. No automatic mm/m conversion is performed.

See [docs/conventions.md](docs/conventions.md) for the complete contract.

## Known limitations

- scalar, constant conductivity and P1 tetra FEM only;
- node-to-DOF mapping and P1 locator are serial/owned-local-cell MVPs, not global MPI ownership maps;
- one Green solve is performed per measurement channel;
- Green functions are retained to build a transfer matrix, which can be memory-heavy;
- central projection checks surface triangles per projected electrode;
- inverse reconstruction supports one point dipole on a discrete candidate set;
- transfer cache provenance is caller-provided and not yet schema-validated;
- solver lifetime requires explicit `destroy()`.
