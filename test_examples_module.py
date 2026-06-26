import importlib.util
from pathlib import Path

import numpy as np
import pytest

from geometry import ElectrodeSet, MeshData


def _load_full_inverse_example():
    module_path = Path(__file__).resolve().parent / "examples" / "full_inverse_experiment_torso.py"
    spec = importlib.util.spec_from_file_location("full_inverse_experiment_torso", module_path)
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
