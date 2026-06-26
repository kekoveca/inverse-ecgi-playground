import numpy as np
import sys
from pathlib import Path

# Allow running as:
#   python3 examples/full_inverse_experiment_torso.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry import (
    MeshData,
    ElectrodeSet,
    SourceRegion,
    TorsoGeometry,
    validate_torso_geometry,
    plot_torso_geometry,
)

# Куб [0,1]x[0,1]x[0,1], разбитый на несколько тетраэдров
points = np.array(
    [
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
    ]
)

# Простое разбиение куба на 5 тетраэдров
cells = np.array(
    [
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 3, 4, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
    ]
)

volume_mesh = MeshData(
    points=points,
    cells=cells,
    cell_type="tetra",
    name="cube_tetra_mesh",
    metadata={"units": "arbitrary"},
)

electrodes = ElectrodeSet(
    positions=np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    ),
    labels=["E1", "E2", "E3", "E4"],
    name="top_surface_electrodes",
)

# Source region через bounding box.
# Выберем ячейки, центры которых лежат внутри центральной области.
source_region = SourceRegion.from_bounding_box(
    mesh=volume_mesh,
    bounds_min=[0.2, 0.2, 0.2],
    bounds_max=[0.8, 0.8, 0.8],
    mode="center",
    name="central_bbox_source_region",
)

geometry = TorsoGeometry(
    geometry_id="cube_demo",
    volume_mesh=volume_mesh,
    electrodes=electrodes,
    source_region=source_region,
    metadata={"description": "Cube split into tetrahedra"},
)

report = validate_torso_geometry(geometry)

print("Is valid:", report.is_valid)
print("Errors:", report.errors)
print("Warnings:", report.warnings)
print("Summary:", geometry.summary())

fig, ax = plot_torso_geometry(
    geometry,
    show_electrodes=True,
    show_source_region=True,
    show_fig=True,
)
