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

The input mesh must expose a tetra physical group named `domain` and a triangle physical group named `boundary` by default. Override them with `--domain-name` and `--boundary-name`.

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

Source/candidate location is performed in DOLFINx ordering by the solver's cached `DOLFINxP1TetraLocator`; MeshData candidate cell ids are not passed through as DOLFINx ids.

Average reference is the default. When using `--reference single`, also pass `--reference-index N`.

## Outputs

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

## ParaView

Open the `.bp` outputs in ParaView. `electrodes.bp` is a diagnostic nodal marker
field: it marks the nearest FEM DOF to each electrode position, with marker
values `1, 2, ...`. It is not an independent point cloud; use
`electrode_marker_mapping.csv` to inspect exact electrode coordinates and
nearest-DOF distances. Use `--no-export` to skip VTX/BP export.

## Notes on electrode placement

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

The production projection path uses a cached `TetraVolumeLocator` for inside checks. Central ray/triangle intersection is a separate operation and can still scale with the number of surface triangles for electrodes that actually require projection.

## Notes on source candidates

The base tutorial selects tetra cell centers from a central bounding-box source region. `SourceRegion.candidate_cell_ids` are MeshData ids; Green transfer locates candidate coordinates again and stores local DOLFINx ids. The true source is chosen by `--source-index` or by proximity to the volume bbox center.

Candidates on shared faces/edges/vertices are mathematically ambiguous for a cell-local P1 dipole gradient. Cell-center candidates avoid that ambiguity.

## Full inverse experiment with clipped-sphere electrodes

`full_inverse_experiment_torso_clipped_sphere_electrodes.py` runs the same
end-to-end pipeline, but builds electrodes from an outer sphere before
projection:

1. create a sphere centered at the geometry bbox center and large enough to
   contain the whole bbox;
2. clip that sphere between two Z planes: bbox `z_min` and `z_max` moved 10%
   toward the bbox center;
3. place `num_electrodes` quasi-uniformly on the remaining spherical band with
   a golden-angle sequence;
4. centrally project those outside points to the torso surface with the
   existing `central_project_electrodes_to_surface` API.

Run:

```bash
python3 examples/full_inverse_experiment_torso_clipped_sphere_electrodes.py \
  --mesh torso.msh \
  --output output/full_inverse_experiment_clipped_sphere \
  --num-electrodes 32 \
  --num-candidates 50 \
  --moment 0 0 1 \
  --lambda-reg 1e-10
```

For a fast no-export smoke run:

```bash
python3 examples/full_inverse_experiment_torso_clipped_sphere_electrodes.py \
  --mesh torso.msh \
  --output output/full_inverse_experiment_clipped_sphere_smoke \
  --num-electrodes 8 \
  --num-candidates 5 \
  --no-export
```

The `--z-trim-fraction` option defaults to `0.1`, matching the 10% inward shift
of the clipping planes.

Both tutorials assume that mesh coordinates, electrode positions, source moments and reported distance thresholds use a consistent coordinate system. No automatic mm/m conversion is performed.

## Troubleshooting

- Missing `domain`/`boundary`: inspect the printed physical groups and pass matching names.
- Empty source region: check coordinate units and source bounding-box placement.
- Green compatibility error: use average reference for the tutorial.
- `projection_surface_cell_id = -1`: the electrode was usually unchanged, not unsuccessfully projected.
- Surface point-array count equals volume count: count unique ids referenced by triangle cells.
- `electrodes.bp` differs from CSV coordinates: the BP field marks nearest FEM dofs by design.
- Slow runs: start with `--num-electrodes 8 --num-candidates 5 --no-export`, then use the profiling scripts.
