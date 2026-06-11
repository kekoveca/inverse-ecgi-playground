from __future__ import annotations

import numpy as np

from geometry import ElectrodeSet, MeshData
from sources import barycentric_coordinates_tetra
from sources import locate_points_in_mesh as _locate_points_in_mesh


def _as_points(points) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (n_points, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")
    return points


def locate_points_in_tetra_mesh(
    mesh: MeshData,
    points,
    candidate_cell_ids=None,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate points in tetrahedra using the shared sources mesh locator.

    ``sources.locate_points_in_mesh`` owns the actual cell search. This function
    adds the barycentric coordinates needed to build electrode interpolation
    rows.
    """
    points = _as_points(points)
    try:
        cell_ids = _locate_points_in_mesh(
            mesh,
            points,
            candidate_cell_ids=candidate_cell_ids,
            tol=tol,
        )
    except ValueError as exc:
        message = str(exc)
        if message.startswith("point "):
            parts = message.split()
            if len(parts) >= 2 and parts[1].isdigit():
                point_id = int(parts[1])
                if point_id < points.shape[0]:
                    coords = np.array2string(points[point_id], precision=16, separator=", ")
                    raise ValueError(
                        f"point {point_id} with coordinates {coords} is not inside any candidate tetrahedron"
                    ) from exc
        raise

    barycentric = np.zeros((points.shape[0], 4), dtype=float)
    for point_id, cell_id in enumerate(cell_ids):
        vertices = mesh.points[mesh.cells[int(cell_id)]]
        barycentric[point_id] = barycentric_coordinates_tetra(points[point_id], vertices)
    return cell_ids.astype(np.int64, copy=False), barycentric


def locate_electrodes_in_mesh(
    mesh: MeshData,
    electrodes: ElectrodeSet,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate electrode positions in a tetrahedral volume mesh."""
    if electrodes.geometric_dim != mesh.geometric_dim:
        raise ValueError("mesh and electrodes must have the same geometric dimension")
    return locate_points_in_tetra_mesh(mesh, electrodes.positions, tol=tol)
