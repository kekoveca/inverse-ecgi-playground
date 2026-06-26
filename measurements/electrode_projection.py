from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from geometry import ElectrodeSet, MeshData
from sources import point_in_tetra


@dataclass(frozen=True)
class ElectrodeProjectionReport:
    """Diagnostics for central projection of electrodes onto a torso surface."""

    original_positions: np.ndarray
    projected_positions: np.ndarray
    projected_mask: np.ndarray
    projection_distances: np.ndarray
    projection_center: np.ndarray
    surface_cell_ids: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        original = np.asarray(self.original_positions, dtype=float)
        projected = np.asarray(self.projected_positions, dtype=float)
        mask = np.asarray(self.projected_mask, dtype=bool)
        distances = np.asarray(self.projection_distances, dtype=float)
        center = np.asarray(self.projection_center, dtype=float)
        surface_cell_ids = np.asarray(self.surface_cell_ids, dtype=np.int64)
        if original.ndim != 2 or original.shape[1] != 3:
            raise ValueError("original_positions must have shape (n_electrodes, 3)")
        if projected.shape != original.shape:
            raise ValueError("projected_positions must match original_positions shape")
        if mask.shape != (original.shape[0],):
            raise ValueError("projected_mask must have one entry per electrode")
        if distances.shape != (original.shape[0],):
            raise ValueError("projection_distances must have one entry per electrode")
        if center.shape != (3,):
            raise ValueError("projection_center must have shape (3,)")
        if surface_cell_ids.shape != (original.shape[0],):
            raise ValueError("surface_cell_ids must have one entry per electrode")
        object.__setattr__(self, "original_positions", original)
        object.__setattr__(self, "projected_positions", projected)
        object.__setattr__(self, "projected_mask", mask)
        object.__setattr__(self, "projection_distances", distances)
        object.__setattr__(self, "projection_center", center)
        object.__setattr__(self, "surface_cell_ids", surface_cell_ids)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def num_projected(self) -> int:
        return int(np.count_nonzero(self.projected_mask))

    @property
    def projected_indices(self) -> np.ndarray:
        return np.flatnonzero(self.projected_mask).astype(np.int64)

    @property
    def max_projection_distance(self) -> float:
        if self.projection_distances.size == 0:
            return 0.0
        return float(self.projection_distances.max())

    def to_summary_dict(self) -> dict[str, Any]:
        """Return JSON-friendly projection diagnostics without large arrays."""
        return {
            "num_electrodes": int(self.projected_mask.size),
            "num_projected": self.num_projected,
            "projected_indices": self.projected_indices.tolist(),
            "max_projection_distance": self.max_projection_distance,
            "projection_center": self.projection_center.tolist(),
            "surface_cell_ids": self.surface_cell_ids.tolist(),
            "metadata": dict(self.metadata),
        }


def _validate_volume_mesh(mesh: MeshData) -> None:
    if mesh.cell_type != "tetra" or mesh.geometric_dim != 3:
        raise ValueError("central electrode projection requires a 3D tetra volume mesh")


def boundary_triangle_mesh_from_tetra_mesh(mesh: MeshData) -> MeshData:
    """Extract a triangle surface mesh from boundary faces of a tetra mesh."""
    _validate_volume_mesh(mesh)
    face_patterns = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )
    faces = mesh.cells[:, face_patterns].reshape(-1, 3)
    sorted_faces = np.sort(faces, axis=1)
    unique_faces, counts = np.unique(sorted_faces, axis=0, return_counts=True)
    boundary_faces = unique_faces[counts == 1]
    if boundary_faces.size == 0:
        raise ValueError("could not infer any boundary triangles from the tetra mesh")
    return MeshData(
        points=mesh.points,
        cells=boundary_faces,
        cell_type="triangle",
        name=f"{mesh.name}_boundary",
        metadata={"source": "boundary_triangle_mesh_from_tetra_mesh", "volume_mesh": mesh.name},
    )


def _surface_mesh(surface_mesh: MeshData | None, volume_mesh: MeshData) -> MeshData:
    if surface_mesh is None:
        return boundary_triangle_mesh_from_tetra_mesh(volume_mesh)
    if surface_mesh.cell_type != "triangle" or surface_mesh.geometric_dim != 3:
        raise ValueError("surface_mesh must be a 3D triangle MeshData")
    return surface_mesh


class TetraVolumeLocator:
    """Cached inside/containing-cell locator for a tetrahedral ``MeshData``.

    The locator owns bbox data, tetra vertices, centroids and one cKDTree. It is
    intended for repeated point-in-volume checks where rebuilding the tree per
    point would dominate runtime.
    """

    def __init__(self, mesh: MeshData, tol: float = 1e-10, initial_k: int = 8) -> None:
        _validate_volume_mesh(mesh)
        if initial_k < 1:
            raise ValueError("initial_k must be positive")
        self.mesh = mesh
        self.tol = float(tol)
        self.initial_k = int(initial_k)
        self.cell_ids = np.arange(mesh.num_cells, dtype=np.int64)
        self.vertices = mesh.points[mesh.cells]
        self.centroids = self.vertices.mean(axis=1)
        self.tree = cKDTree(self.centroids)
        self.bbox_min, self.bbox_max = mesh.bounding_box()

    def _as_points(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (n_points, 3), got {points.shape}")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must contain only finite values")
        return points

    def locate_points(self, points) -> np.ndarray:
        """Return containing cell ids or ``-1`` for points outside the volume."""
        points = self._as_points(points)
        located = np.full(points.shape[0], -1, dtype=np.int64)
        if points.shape[0] == 0 or self.cell_ids.size == 0:
            return located

        bbox_tol = self.tol
        inside_bbox = np.all(points >= self.bbox_min - bbox_tol, axis=1) & np.all(
            points <= self.bbox_max + bbox_tol,
            axis=1,
        )
        if not np.any(inside_bbox):
            return located

        unresolved = np.flatnonzero(inside_bbox)
        k = min(self.initial_k, self.cell_ids.size)
        while unresolved.size > 0:
            _, local_indices = self.tree.query(points[unresolved], k=k)
            if k == 1:
                local_indices = local_indices[:, np.newaxis]

            still_unresolved: list[int] = []
            for row, point_id in enumerate(unresolved):
                point = points[point_id]
                for local_cell_id in np.atleast_1d(local_indices[row]):
                    cell_id = int(self.cell_ids[int(local_cell_id)])
                    if point_in_tetra(point, self.vertices[cell_id], tol=self.tol):
                        located[point_id] = cell_id
                        break
                if located[point_id] < 0:
                    still_unresolved.append(int(point_id))

            if not still_unresolved or k == self.cell_ids.size:
                break
            unresolved = np.asarray(still_unresolved, dtype=np.int64)
            k = min(2 * k, self.cell_ids.size)

        return located

    def contains_points(self, points) -> np.ndarray:
        """Return a boolean mask for points inside or on the tetra volume."""
        return self.locate_points(points) >= 0

    def contains_point(self, point) -> bool:
        """Return whether one point is inside or on the tetra volume."""
        return bool(self.contains_points(np.asarray(point, dtype=float).reshape(1, 3))[0])


class CentralSurfaceProjector:
    """Cached central ray projector onto a triangular surface mesh."""

    def __init__(self, surface_mesh: MeshData, center, tol: float = 1e-10) -> None:
        if surface_mesh.cell_type != "triangle" or surface_mesh.geometric_dim != 3:
            raise ValueError("surface_mesh must be a 3D triangle MeshData")
        center = np.asarray(center, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("projection center must have shape (3,) and contain finite values")
        self.surface_mesh = surface_mesh
        self.center = center
        self.tol = float(tol)
        self.triangles = np.asarray(surface_mesh.points[surface_mesh.cells], dtype=float)
        if self.triangles.size == 0:
            raise ValueError("surface_mesh must contain at least one triangle")

    def project_point(self, point) -> tuple[np.ndarray, int]:
        """Project one point from ``center`` to the first surface hit."""
        point = np.asarray(point, dtype=float)
        if point.shape != (3,):
            raise ValueError("point must have shape (3,)")
        if not np.all(np.isfinite(point)):
            raise ValueError("point must contain only finite values")

        direction = point - self.center
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= self.tol:
            raise ValueError("cannot centrally project an electrode located at the projection center")

        best_t = None
        best_cell_id = None
        for cell_id, triangle in enumerate(self.triangles):
            ray_t = _ray_triangle_intersection(self.center, direction, triangle, tol=self.tol)
            if ray_t is None:
                continue
            if best_t is None or ray_t < best_t:
                best_t = float(ray_t)
                best_cell_id = int(cell_id)
        if best_t is None or best_cell_id is None:
            raise ValueError(
                f"central projection ray from {self.center.tolist()} through {point.tolist()} missed the surface"
            )
        return self.center + best_t * direction, best_cell_id

    def project_points(self, points) -> tuple[np.ndarray, np.ndarray]:
        """Project multiple points and return projected positions plus cell ids."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (n_points, 3), got {points.shape}")
        projected = np.zeros_like(points)
        surface_cell_ids = np.full(points.shape[0], -1, dtype=np.int64)
        for point_id, point in enumerate(points):
            projected[point_id], surface_cell_ids[point_id] = self.project_point(point)
        return projected, surface_cell_ids


def _is_inside_volume(mesh: MeshData, point: np.ndarray, tol: float) -> bool:
    return TetraVolumeLocator(mesh, tol=tol).contains_point(point)


def _ray_triangle_intersection(origin: np.ndarray, direction: np.ndarray, triangle: np.ndarray, tol: float) -> float | None:
    v0, v1, v2 = triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    a = float(np.dot(edge1, h))
    if abs(a) <= tol:
        return None
    f = 1.0 / a
    s = origin - v0
    u = f * float(np.dot(s, h))
    if u < -tol or u > 1.0 + tol:
        return None
    q = np.cross(s, edge1)
    v = f * float(np.dot(direction, q))
    if v < -tol or u + v > 1.0 + tol:
        return None
    ray_t = f * float(np.dot(edge2, q))
    if ray_t <= tol:
        return None
    return ray_t


def central_project_point_to_surface(
    point,
    surface_mesh: MeshData,
    center,
    tol: float = 1e-10,
) -> tuple[np.ndarray, int]:
    """Project one point from ``center`` onto the first surface hit by a ray."""
    projector = CentralSurfaceProjector(surface_mesh, center=center, tol=tol)
    return projector.project_point(point)


def central_project_electrodes_to_surface(
    volume_mesh: MeshData,
    electrodes: ElectrodeSet,
    surface_mesh: MeshData | None = None,
    center=None,
    tol: float = 1e-10,
    *,
    project_only_outside: bool = True,
    volume_locator: TetraVolumeLocator | None = None,
    surface_projector: CentralSurfaceProjector | None = None,
) -> tuple[ElectrodeSet, ElectrodeProjectionReport]:
    """Project electrodes outside a tetra volume onto the torso surface.

    By default, electrodes already inside or on the volume mesh are left
    unchanged. Outside electrodes are projected along the ray from ``center``
    through the electrode to the first triangle of ``surface_mesh``. If no
    surface is supplied, the boundary triangles are inferred from the tetra
    volume mesh. Locator/projector objects may be supplied to reuse cached
    spatial data across calls.
    """
    _validate_volume_mesh(volume_mesh)
    if electrodes.geometric_dim != volume_mesh.geometric_dim:
        raise ValueError("mesh and electrodes must have the same geometric dimension")
    surface = _surface_mesh(surface_mesh, volume_mesh)
    if center is None:
        center = volume_mesh.points.mean(axis=0)
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("projection center must have shape (3,) and contain finite values")
    if volume_locator is not None and volume_locator.mesh is not volume_mesh:
        raise ValueError("volume_locator must be built from the same volume_mesh object")
    if surface_projector is not None and surface_projector.surface_mesh is not surface:
        raise ValueError("surface_projector must be built from the same surface_mesh object")
    if surface_projector is not None and not np.allclose(surface_projector.center, center):
        raise ValueError("surface_projector center does not match requested projection center")

    if project_only_outside and volume_locator is None:
        volume_locator = TetraVolumeLocator(volume_mesh, tol=tol)
    if surface_projector is None:
        surface_projector = CentralSurfaceProjector(surface, center=center, tol=tol)

    projected_positions = electrodes.positions.copy()
    projected_mask = np.zeros(electrodes.num_electrodes, dtype=bool)
    projection_distances = np.zeros(electrodes.num_electrodes, dtype=float)
    surface_cell_ids = np.full(electrodes.num_electrodes, -1, dtype=np.int64)

    if project_only_outside:
        assert volume_locator is not None
        inside_mask = volume_locator.contains_points(electrodes.positions)
    else:
        inside_mask = np.zeros(electrodes.num_electrodes, dtype=bool)

    for electrode_id, position in enumerate(electrodes.positions):
        if inside_mask[electrode_id]:
            continue
        projected, surface_cell_id = surface_projector.project_point(position)
        projected_positions[electrode_id] = projected
        projected_mask[electrode_id] = True
        projection_distances[electrode_id] = float(np.linalg.norm(position - projected))
        surface_cell_ids[electrode_id] = int(surface_cell_id)

    metadata = {
        "projection": "central",
        "surface_mesh": surface.name,
        "only_outside_electrodes": bool(project_only_outside),
        "volume_locator": None if not project_only_outside else "TetraVolumeLocator",
        "surface_projector": "CentralSurfaceProjector",
    }
    report = ElectrodeProjectionReport(
        original_positions=electrodes.positions,
        projected_positions=projected_positions,
        projected_mask=projected_mask,
        projection_distances=projection_distances,
        projection_center=center,
        surface_cell_ids=surface_cell_ids,
        metadata=metadata,
    )
    projected_electrodes = ElectrodeSet(
        positions=projected_positions,
        labels=electrodes.labels,
        name=electrodes.name,
        metadata={
            **electrodes.metadata,
            "projection": report.to_summary_dict(),
        },
    )
    return projected_electrodes, report
