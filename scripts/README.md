# Profiling scripts

Run scripts from the repository root.

## Full inverse profile

```bash
python3 scripts/profile_full_inverse_experiment.py \
  --mesh meshes/torso_refined.msh \
  --output output/performance_profile \
  --num-electrodes 128 \
  --num-candidates 50 \
  --max-green-rows 8 \
  --no-export
```

This requires the DOLFINx runtime and a Gmsh mesh with `domain` and `boundary` physical groups. Outputs are `timing.csv`, `timing.json`, `memory.json` and `profile_summary.md`.

Average reference is the default. `--reference single` requires `--reference-index N`.

## Component profiles

```bash
python3 scripts/profile_components.py --component point-location --mesh meshes/torso_refined.msh
python3 scripts/profile_components.py --component green-transfer --mesh meshes/torso_refined.msh
python3 scripts/profile_components.py --component inverse-scaling
```

Without `--mesh`, DOLFINx component profiles use a generated unit-cube `MeshData`. `inverse-scaling` is numpy-only.

See [../docs/performance.md](../docs/performance.md) for stage definitions, outputs and interpretation.
