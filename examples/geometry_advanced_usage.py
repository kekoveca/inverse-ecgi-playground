import numpy as np
import sys

from pathlib import Path

# Allow running as:
#   python3 examples/full_inverse_experiment_torso.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry import (
    read_gmsh_meshio,
    ElectrodeSet,
    SourceRegion,
    TorsoGeometry,
    validate_torso_geometry,
    plot_torso_geometry,
)

mesh = read_gmsh_meshio("torso.msh", dim=3)

volume_mesh = mesh.to_mesh_data(
    cell_type="tetra",
    physical_name="domain",
)

print(volume_mesh.bounding_box())

surface_mesh = mesh.to_mesh_data(
    cell_type="triangle",
    physical_name="boundary",
)

print(volume_mesh.num_cells)  # должно быть 47158
print(surface_mesh.num_cells)  # должно быть 7760

print("physical tag domain:", mesh.physical_tag("domain"))

tetra_all = mesh.cell_block("tetra")
tetra_domain = mesh.cell_block("tetra", physical_name="domain")

print("tetra_all:", tetra_all.shape)
print("tetra_domain:", tetra_domain.shape)

print("unique tetra tags:", np.unique(mesh.cell_tags["tetra"]))

electrodes = ElectrodeSet(
    positions=np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    ),
    labels=["E1", "E2"],
)

source_region = SourceRegion.from_bounding_box(
    mesh=volume_mesh,
    bounds_min=[-50, -50, -50],
    bounds_max=[50, 50, 50],
    mode="center",
)

geometry = TorsoGeometry(
    geometry_id="torso_from_msh",
    volume_mesh=volume_mesh,
    surface_mesh=surface_mesh,
    electrodes=electrodes,
    source_region=source_region,
)

report = validate_torso_geometry(geometry)

print(report.is_valid)
print(report.errors)
print(report.warnings)
print(geometry.summary())

plot_torso_geometry(
    geometry,
    show_source_region=True,
    show_fig=True,
)
