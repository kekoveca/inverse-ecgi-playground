import numpy as np

from geometry import (
    read_gmsh_meshio,
    ElectrodeSet,
    SourceRegion,
    TorsoGeometry,
    validate_torso_geometry,
    plot_torso_geometry,
)

tagged = read_gmsh_meshio("torso.msh", dim=3)

volume_mesh = tagged.to_mesh_data(
    cell_type="tetra",
    physical_name="domain",
)

print(volume_mesh.bounding_box())

surface_mesh = tagged.to_mesh_data(
    cell_type="triangle",
    physical_name="boundary",
)

print(volume_mesh.num_cells)  # должно быть 47158
print(surface_mesh.num_cells)  # должно быть 7760

print("physical tag domain:", tagged.physical_tag("domain"))

tetra_all = tagged.cell_block("tetra")
tetra_domain = tagged.cell_block("tetra", physical_name="domain")

print("tetra_all:", tetra_all.shape)
print("tetra_domain:", tetra_domain.shape)

print("unique tetra tags:", np.unique(tagged.cell_tags["tetra"]))

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
    bounds_min=[-0.1, -0.1, -0.1],
    bounds_max=[0.1, 0.1, 0.1],
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
