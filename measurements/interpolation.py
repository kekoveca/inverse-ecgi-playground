from __future__ import annotations

import warnings

import numpy as np

from geometry import MeshData

from .electrode_locator import locate_points_in_tetra_mesh


def _as_points(points) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (n_points, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")
    return points


def _as_location_data(mesh: MeshData, points, cell_ids, barycentric, tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = _as_points(points)
    if cell_ids is None or barycentric is None:
        if cell_ids is not None or barycentric is not None:
            raise ValueError("cell_ids and barycentric must be provided together")
        cell_ids, barycentric = locate_points_in_tetra_mesh(mesh, points, tol=tol)
    else:
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        barycentric = np.asarray(barycentric, dtype=float)

    if cell_ids.shape != (points.shape[0],):
        raise ValueError(f"cell_ids must have shape ({points.shape[0]},), got {cell_ids.shape}")
    if barycentric.shape != (points.shape[0], 4):
        raise ValueError(f"barycentric must have shape ({points.shape[0]}, 4), got {barycentric.shape}")
    if cell_ids.size > 0 and (cell_ids.min() < 0 or cell_ids.max() >= mesh.num_cells):
        raise ValueError("cell_ids contain ids outside mesh cells")
    return points, cell_ids, barycentric


def build_point_interpolation_matrix(
    mesh: MeshData,
    points,
    cell_ids=None,
    barycentric=None,
    tol: float = 1e-10,
    sparse: bool = True,
):
    """Build the P1 interpolation matrix from MeshData nodes to points.

    Each tetrahedral row has four barycentric weights. Input nodal vectors must
    use MeshData node ordering; DOLFINx DOF ordering requires a verified map.
    """
    points, cell_ids, barycentric = _as_location_data(mesh, points, cell_ids, barycentric, tol)
    n_points = points.shape[0]
    n_nodes = mesh.num_points
    nodes = mesh.cells[cell_ids]

    if sparse:
        try:
            from scipy.sparse import csr_matrix
        except ImportError:  # pragma: no cover - scipy is available in tests
            warnings.warn(
                "scipy is not available; returning a dense interpolation matrix",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            row_indices = np.repeat(np.arange(n_points, dtype=np.int64), 4)
            col_indices = nodes.reshape(-1)
            data = barycentric.reshape(-1)
            return csr_matrix((data, (row_indices, col_indices)), shape=(n_points, n_nodes))

    matrix = np.zeros((n_points, n_nodes), dtype=float)
    for point_id in range(n_points):
        matrix[point_id, nodes[point_id]] = barycentric[point_id]
    return matrix


def evaluate_at_points(
    mesh: MeshData,
    nodal_values,
    points,
    cell_ids=None,
    barycentric=None,
    tol: float = 1e-10,
) -> np.ndarray:
    """Evaluate nodal P1 values at arbitrary points inside tetrahedra."""
    values = np.asarray(nodal_values, dtype=float)
    if values.shape != (mesh.num_points,):
        raise ValueError(f"nodal_values must have shape ({mesh.num_points},), got {values.shape}")
    matrix = build_point_interpolation_matrix(
        mesh,
        points,
        cell_ids=cell_ids,
        barycentric=barycentric,
        tol=tol,
        sparse=False,
    )
    return np.asarray(matrix @ values, dtype=float)
