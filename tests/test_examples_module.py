import importlib.util
from pathlib import Path

import numpy as np
import pytest

from geometry import ElectrodeSet, MeshData


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_full_inverse_example():
    module_path = PROJECT_ROOT / "examples" / "full_inverse_experiment_torso.py"
    spec = importlib.util.spec_from_file_location("full_inverse_experiment_torso", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_clipped_sphere_example():
    module_path = (
        PROJECT_ROOT
        / "examples"
        / "full_inverse_experiment_torso_clipped_sphere_electrodes.py"
    )
    spec = importlib.util.spec_from_file_location("full_inverse_experiment_torso_clipped_sphere_electrodes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_surface_used_vertices_count():
    example = _load_full_inverse_example()
    surface = MeshData(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [9.0, 9.0, 9.0],
            ]
        ),
        cells=np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype=np.int64,
        ),
        cell_type="triangle",
    )

    used = example.surface_used_vertex_ids(surface)

    assert np.array_equal(used, np.array([0, 1, 2, 3]))
    assert used.size == 4


def test_inspect_electrodes_nearest_surface_cells_single_triangle():
    example = _load_full_inverse_example()
    surface = MeshData(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        cells=np.array([[0, 1, 2]], dtype=np.int64),
        cell_type="triangle",
    )
    electrodes = ElectrodeSet(
        positions=np.array([[0.0, 0.0, 0.0]]),
        labels=["E1"],
    )

    info = example.inspect_electrodes_nearest_surface_cells(surface, electrodes)

    assert info["num_electrodes"] == 1
    assert np.array_equal(info["nearest_surface_cell_ids"], np.array([0]))
    assert info["nearest_surface_distances"][0] == pytest.approx(0.0, abs=1e-12)
    assert info["max_nearest_surface_distance"] == pytest.approx(0.0, abs=1e-12)
    assert info["mean_nearest_surface_distance"] == pytest.approx(0.0, abs=1e-12)


def test_clipped_bbox_sphere_parameters():
    example = _load_clipped_sphere_example()
    mesh = MeshData(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 6.0],
                [2.0, 4.0, 6.0],
            ]
        ),
        cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
        cell_type="tetra",
    )

    params = example.clipped_bbox_sphere_parameters(mesh, z_trim_fraction=0.1)

    assert np.allclose(params["center"], [1.0, 2.0, 3.0])
    assert params["radius"] == pytest.approx(np.sqrt(14.0))
    assert params["z_clip_min"] == pytest.approx(0.3)
    assert params["z_clip_max"] == pytest.approx(5.7)


def test_quasiuniform_points_on_clipped_sphere():
    example = _load_clipped_sphere_example()
    center = np.array([1.0, 2.0, 3.0])
    radius = 5.0

    points = example.quasiuniform_points_on_clipped_sphere(
        center=center,
        radius=radius,
        z_clip_min=1.0,
        z_clip_max=5.0,
        num_points=12,
        seed=42,
    )

    assert points.shape == (12, 3)
    assert np.all(points[:, 2] >= 1.0)
    assert np.all(points[:, 2] <= 5.0)
    assert np.allclose(np.linalg.norm(points - center, axis=1), radius)
