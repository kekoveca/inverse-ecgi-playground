from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mesh_model import MeshData


@dataclass(frozen=True)
class SourceRegion:
    """Candidate region where a point dipole may be located.

    The first implementation represents a source region as a finite list of
    candidate points and the ids of the mesh cells that generated them.
    For P1 FEM this is convenient because Green gradients are constant on each
    tetrahedron/triangle cell.
    """

    candidate_points: np.ndarray
    candidate_cell_ids: np.ndarray
    name: str = "source_region"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.candidate_points, dtype=float)
        cell_ids = np.asarray(self.candidate_cell_ids, dtype=np.int64)

        if points.ndim != 2:
            raise ValueError("candidate_points must have shape (n_candidates, geometric_dim)")
        if points.shape[1] not in (2, 3):
            raise ValueError("candidate_points must contain 2D or 3D coordinates")
        if cell_ids.ndim != 1:
            raise ValueError("candidate_cell_ids must have shape (n_candidates,)")
        if points.shape[0] != cell_ids.shape[0]:
            raise ValueError("candidate_points and candidate_cell_ids must have the same length")

        object.__setattr__(self, "candidate_points", points)
        object.__setattr__(self, "candidate_cell_ids", cell_ids)

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_points.shape[0])

    @property
    def geometric_dim(self) -> int:
        return int(self.candidate_points.shape[1])

    @classmethod
    def from_cell_ids(
        cls,
        mesh: MeshData,
        cell_ids: np.ndarray,
        name: str = "source_region",
        metadata: dict[str, Any] | None = None,
    ) -> "SourceRegion":
        """Build candidates from mesh cell centers for selected cells."""
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        if cell_ids.ndim != 1:
            raise ValueError("cell_ids must be one-dimensional")
        if cell_ids.size > 0:
            if cell_ids.min() < 0 or cell_ids.max() >= mesh.num_cells:
                raise ValueError("cell_ids contain invalid mesh cell indices")
        centers = mesh.cell_centers(cell_ids)
        return cls(candidate_points=centers, candidate_cell_ids=cell_ids, name=name, metadata=metadata or {})

    @classmethod
    def all_cells(cls, mesh: MeshData, name: str = "all_cells_source_region") -> "SourceRegion":
        """Use all mesh cell centers as source candidates."""
        return cls.from_cell_ids(mesh, np.arange(mesh.num_cells, dtype=np.int64), name=name)

    @classmethod
    def from_bounding_box(
        cls,
        mesh: MeshData,
        bounds_min: np.ndarray | list[float] | tuple[float, ...],
        bounds_max: np.ndarray | list[float] | tuple[float, ...],
        name: str = "bbox_source_region",
        mode: str = "center",
        metadata: dict[str, Any] | None = None,
    ) -> "SourceRegion":
        """Build a source region from cells selected by an axis-aligned bounding box.

        Parameters
        ----------
        mesh:
            Mesh whose cells will be filtered.
        bounds_min, bounds_max:
            Lower and upper corners of the bounding box. Their length must match
            ``mesh.geometric_dim``.
        name:
            Name of the resulting source region.
        mode:
            Cell selection rule:

            - ``"center"``: include cells whose center lies inside the box.
            - ``"any_vertex"``: include cells with at least one vertex inside the box.
            - ``"all_vertices"``: include cells whose all vertices lie inside the box.

        Returns
        -------
        SourceRegion
            Candidate points are selected cell centers. Candidate cell ids refer
            to the original cells in ``mesh``.
        """
        bounds_min_arr = np.asarray(bounds_min, dtype=float)
        bounds_max_arr = np.asarray(bounds_max, dtype=float)

        if bounds_min_arr.shape != (mesh.geometric_dim,):
            raise ValueError(f"bounds_min must have shape ({mesh.geometric_dim},)")
        if bounds_max_arr.shape != (mesh.geometric_dim,):
            raise ValueError(f"bounds_max must have shape ({mesh.geometric_dim},)")
        if np.any(bounds_min_arr > bounds_max_arr):
            raise ValueError("bounds_min must be <= bounds_max component-wise")

        if mode == "center":
            centers = mesh.cell_centers()
            mask = np.all((centers >= bounds_min_arr) & (centers <= bounds_max_arr), axis=1)
        elif mode in {"any_vertex", "all_vertices"}:
            cell_points = mesh.points[mesh.cells]
            vertex_inside = np.all((cell_points >= bounds_min_arr) & (cell_points <= bounds_max_arr), axis=2)
            if mode == "any_vertex":
                mask = np.any(vertex_inside, axis=1)
            else:
                mask = np.all(vertex_inside, axis=1)
        else:
            raise ValueError("mode must be one of: 'center', 'any_vertex', 'all_vertices'")

        cell_ids = np.flatnonzero(mask).astype(np.int64)
        region_metadata = dict(metadata or {})
        region_metadata.update(
            {
                "selection": "bounding_box",
                "bounds_min": bounds_min_arr.tolist(),
                "bounds_max": bounds_max_arr.tolist(),
                "mode": mode,
                "source_mesh_name": mesh.name,
            }
        )
        return cls.from_cell_ids(mesh=mesh, cell_ids=cell_ids, name=name, metadata=region_metadata)

    def subset(self, ids: np.ndarray, name: str | None = None) -> "SourceRegion":
        ids = np.asarray(ids, dtype=np.int64)
        return SourceRegion(
            candidate_points=self.candidate_points[ids],
            candidate_cell_ids=self.candidate_cell_ids[ids],
            name=name or self.name,
            metadata=dict(self.metadata),
        )
