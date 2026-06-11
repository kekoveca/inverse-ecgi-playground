from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from geometry import MeshData

from .cell_geometry import gradients_p1_tetra, point_in_tetra
from .point_dipole import PointDipole


def _validate_tetra_mesh(mesh: MeshData) -> None:
    if mesh.cell_type != "tetra":
        raise ValueError("point dipole RHS assembly requires mesh.cell_type='tetra'")
    if mesh.geometric_dim != 3:
        raise ValueError("point dipole RHS assembly requires 3D mesh points")
    if mesh.cells.shape[1] != 4:
        raise ValueError("tetra mesh cells must have shape (n_cells, 4)")


def locate_point_in_mesh(
    mesh: MeshData,
    point: np.ndarray,
    candidate_cell_ids=None,
    tol: float = 1e-10,
) -> int:
    """Return the first tetrahedron id containing ``point``."""
    return int(locate_points_in_mesh(mesh, np.asarray(point, dtype=float).reshape(1, -1), candidate_cell_ids, tol)[0])


def locate_points_in_mesh(
    mesh: MeshData,
    points: np.ndarray,
    candidate_cell_ids=None,
    tol: float = 1e-10,
    initial_k: int = 8,
) -> np.ndarray:
    """Locate points in tetrahedral cells using a cKDTree over cell centroids.

    The KD-tree is used only to order likely cells. A point is accepted only
    after the barycentric ``point_in_tetra`` check succeeds, so the result is
    still geometric rather than nearest-centroid based.
    """
    _validate_tetra_mesh(mesh)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (n_points, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    if candidate_cell_ids is None:
        cell_ids = np.arange(mesh.num_cells, dtype=np.int64)
    else:
        cell_ids = np.asarray(candidate_cell_ids, dtype=np.int64)
        if cell_ids.ndim != 1:
            raise ValueError("candidate_cell_ids must be one-dimensional")
    if cell_ids.size == 0:
        raise ValueError("candidate_cell_ids must contain at least one cell")
    if cell_ids.min() < 0 or cell_ids.max() >= mesh.num_cells:
        raise ValueError("candidate_cell_ids contain ids outside mesh cells")

    if initial_k < 1:
        raise ValueError("initial_k must be positive")

    centroids = mesh.points[mesh.cells[cell_ids]].mean(axis=1)
    tree = cKDTree(centroids)
    located = np.full(points.shape[0], -1, dtype=np.int64)

    k = min(int(initial_k), cell_ids.size)
    while np.any(located < 0):
        unresolved = np.flatnonzero(located < 0)
        _, local_indices = tree.query(points[unresolved], k=k)
        if k == 1:
            local_indices = local_indices[:, np.newaxis]

        for row, point_id in enumerate(unresolved):
            point = points[point_id]
            for local_cell_id in np.atleast_1d(local_indices[row]):
                cell_id = int(cell_ids[int(local_cell_id)])
                vertices = mesh.points[mesh.cells[cell_id]]
                if point_in_tetra(point, vertices, tol=tol):
                    located[point_id] = cell_id
                    break

        if not np.any(located < 0):
            break
        if k == cell_ids.size:
            first_missing = int(np.flatnonzero(located < 0)[0])
            raise ValueError(f"point {first_missing} is not inside any candidate tetrahedron")
        k = min(2 * k, cell_ids.size)

    return located


def _resolve_cell_id(mesh: MeshData, source: PointDipole, cell_id: int | None) -> int:
    if cell_id is not None:
        resolved = int(cell_id)
    elif source.cell_id is not None:
        resolved = int(source.cell_id)
    else:
        resolved = locate_point_in_mesh(mesh, source.position)
    if resolved < 0 or resolved >= mesh.num_cells:
        raise ValueError(f"cell_id {resolved} is outside mesh cells")
    return resolved


def assemble_point_dipole_rhs_numpy(mesh: MeshData, source: PointDipole, cell_id: int | None = None) -> np.ndarray:
    """Assemble the nodal RHS for a P1 point dipole on a tetrahedral mesh."""
    _validate_tetra_mesh(mesh)
    resolved_cell_id = _resolve_cell_id(mesh, source, cell_id)

    global_dofs = mesh.cells[resolved_cell_id]
    vertices = mesh.points[global_dofs]
    grads = gradients_p1_tetra(vertices)
    local_rhs = grads @ source.moment

    rhs = np.zeros(mesh.num_points, dtype=float)
    np.add.at(rhs, global_dofs, local_rhs)
    return rhs


def rhs_compatibility_error(rhs) -> float:
    rhs = np.asarray(rhs, dtype=float)
    return abs(float(rhs.sum()))


def check_rhs_compatibility(rhs, tol: float = 1e-10) -> bool:
    rhs = np.asarray(rhs, dtype=float)
    norm = float(np.linalg.norm(rhs))
    scale = max(1.0, norm)
    return rhs_compatibility_error(rhs) <= tol * scale


def assemble_point_dipole_rhs_petsc(solver, source: PointDipole, cell_id: int | None = None):
    """Assemble a point dipole RHS object compatible with ``solver.solve``."""
    if not hasattr(solver, "mesh_data"):
        raise TypeError("solver must expose mesh_data")
    if not hasattr(solver, "rhs_from_local_array"):
        raise TypeError("solver must provide rhs_from_local_array(values)")

    rhs = assemble_point_dipole_rhs_numpy(solver.mesh_data, source, cell_id=cell_id)
    return solver.rhs_from_local_array(rhs)
