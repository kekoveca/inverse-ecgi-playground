from __future__ import annotations

import numpy as np


def _as_vertices(vertices) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (4, 3):
        raise ValueError(f"vertices must have shape (4, 3), got {vertices.shape}")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must contain only finite values")
    return vertices


def _as_point(point) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"point must have shape (3,), got {point.shape}")
    if not np.all(np.isfinite(point)):
        raise ValueError("point must contain only finite values")
    return point


def tetra_signed_volume(vertices) -> float:
    """Return signed tetrahedron volume."""
    vertices = _as_vertices(vertices)
    v0, v1, v2, v3 = vertices
    return float(np.linalg.det(np.column_stack((v1 - v0, v2 - v0, v3 - v0))) / 6.0)


def tetra_volume(vertices) -> float:
    """Return absolute tetrahedron volume."""
    return abs(tetra_signed_volume(vertices))


def _barycentric_matrix(vertices: np.ndarray, *, tol: float) -> np.ndarray:
    matrix = np.vstack((vertices.T, np.ones(4)))
    try:
        inv_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("degenerate tetrahedron: barycentric matrix is singular") from exc

    volume = tetra_volume(vertices)
    if volume <= tol:
        raise ValueError(f"degenerate tetrahedron: volume={volume:g}")
    return inv_matrix


def barycentric_coordinates_tetra(point, vertices, tol: float = 1e-14) -> np.ndarray:
    """Return barycentric coordinates of ``point`` in a tetrahedron."""
    vertices = _as_vertices(vertices)
    point = _as_point(point)
    inv_matrix = _barycentric_matrix(vertices, tol=tol)
    rhs = np.array([point[0], point[1], point[2], 1.0], dtype=float)
    return inv_matrix @ rhs


def barycentric_boundary_flags(barycentric, tol: float = 1e-10) -> dict:
    """Classify whether barycentric coordinates are near a face/edge/vertex.

    Interior points have no coordinate near 0 or 1. Points near tetrahedron
    boundaries are geometrically ambiguous for cell-local P1 dipole assembly,
    because multiple neighboring cells may be valid containing cells.
    """
    lambdas = np.asarray(barycentric, dtype=float)
    if lambdas.shape != (4,):
        raise ValueError(f"barycentric must have shape (4,), got {lambdas.shape}")
    if not np.all(np.isfinite(lambdas)):
        raise ValueError("barycentric coordinates must be finite")
    tol = float(tol)
    near_zero = np.flatnonzero(np.abs(lambdas) <= tol).astype(np.int64)
    near_one = np.flatnonzero(np.abs(lambdas - 1.0) <= tol).astype(np.int64)
    if near_one.size > 0:
        boundary_kind = "vertex"
    elif near_zero.size >= 2:
        boundary_kind = "edge"
    elif near_zero.size == 1:
        boundary_kind = "face"
    else:
        boundary_kind = "interior"
    is_on_boundary = boundary_kind != "interior"
    return {
        "is_on_boundary": bool(is_on_boundary),
        "boundary_kind": boundary_kind,
        "near_zero_indices": near_zero,
        "near_one_indices": near_one,
        "min_barycentric": float(lambdas.min()),
        "max_barycentric": float(lambdas.max()),
    }


def point_in_tetra(point, vertices, tol: float = 1e-10) -> bool:
    """Return whether ``point`` lies inside or on a tetrahedron."""
    lambdas = barycentric_coordinates_tetra(point, vertices, tol=min(tol, 1e-14))
    return bool(np.all(lambdas >= -tol) and np.all(lambdas <= 1.0 + tol) and abs(lambdas.sum() - 1.0) <= tol)


def gradients_p1_tetra(vertices, tol: float = 1e-14) -> np.ndarray:
    """Return gradients of the four P1 basis functions on a tetrahedron."""
    vertices = _as_vertices(vertices)
    inv_matrix = _barycentric_matrix(vertices, tol=tol)
    return inv_matrix[:, :3].copy()
