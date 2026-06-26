# Examples

## Full inverse experiment on `torso.msh`

`full_inverse_experiment_torso.py` is a tutorial-style end-to-end script:

```text
import torso.msh
  -> extract volume/surface physical groups
  -> select demo electrodes on the surface
  -> run existing central electrode projection API
  -> create source candidates
  -> solve forward FEM problem
  -> solve Green problems
  -> build GreenTransferMatrix
  -> solve single-dipole inverse
  -> save reports and optional ParaView exports
```

Run:

```bash
python3 examples/full_inverse_experiment_torso.py \
  --mesh torso.msh \
  --output output/full_inverse_experiment \
  --num-electrodes 32 \
  --num-candidates 50 \
  --moment 0 0 1 \
  --snr-db 40 \
  --lambda-reg 1e-10
```

For a faster smoke run without ParaView files:

```bash
python3 examples/full_inverse_experiment_torso.py \
  --mesh torso.msh \
  --output output/full_inverse_experiment_smoke \
  --num-electrodes 8 \
  --num-candidates 5 \
  --no-export
```

The example uses the existing project API:

- `read_gmsh_meshio` for Gmsh import;
- `central_project_electrodes_to_surface` for electrode projection;
- `NeumannPoissonSolver` and `ForwardSolver` for the direct problem;
- `GreenSolver` and `build_green_transfer_matrix` for Green transfer;
- `SingleDipoleInverseSolver` for inverse reconstruction.

Outputs:

```text
experiment_summary.json
inverse_summary.json
measurements.npz
electrodes.csv
electrode_surface_diagnostics.csv
electrode_marker_mapping.csv
candidates.csv
electrodes.bp
potential.bp
rhs.bp
true_source_marker.bp
estimated_source_marker.bp
```

Open the `.bp` outputs in ParaView. `electrodes.bp` is a diagnostic nodal marker
field: it marks the nearest FEM DOF to each electrode position, with marker
values `1, 2, ...`. It is not an independent point cloud; use
`electrode_marker_mapping.csv` to inspect exact electrode coordinates and
nearest-DOF distances. Use `--no-export` to skip VTX/BP export.

## Surface diagnostics

The example prints both `surface_mesh` point-array size and
`num_surface_used_vertices`. Some mesh import paths keep the full global point
array on the extracted surface mesh, so the surface point-array size can match
the volume point-array size even though boundary triangles use only a subset of
those points.

`projection_surface_cell_id = -1` in diagnostics means the production central
projection API did not record a projection triangle for that electrode. This is
usually expected when `only_outside_electrodes=True` and the selected electrode
was already on or inside the volume. The separate `nearest_surface_cell_id` is
computed for every electrode by the example and saved in
`electrode_surface_diagnostics.csv`.
