from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .electrodes import ElectrodeSet
from .mesh_model import MeshData
from .source_region import SourceRegion
from .torso_geometry import TorsoGeometry


@dataclass(frozen=True)
class AffineTransform:
    matrix: np.ndarray
    offset: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        offset = np.asarray(self.offset, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        if offset.shape != (matrix.shape[0],):
            raise ValueError("offset must have shape (dim,)")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)

    @classmethod
    def identity(cls, dim: int) -> "AffineTransform":
        return cls(matrix=np.eye(dim), offset=np.zeros(dim))

    @classmethod
    def scale(cls, factors: float | np.ndarray, dim: int = 3) -> "AffineTransform":
        f = np.asarray(factors, dtype=float)
        if f.ndim == 0:
            f = np.full(dim, float(f))
        if f.shape != (dim,):
            raise ValueError("scale factors must be scalar or shape (dim,)")
        return cls(matrix=np.diag(f), offset=np.zeros(dim))

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float) @ self.matrix.T + self.offset


def transform_mesh(mesh: MeshData, transform: AffineTransform, name_suffix: str = "_affine") -> MeshData:
    return MeshData(
        points=transform.apply_points(mesh.points),
        cells=mesh.cells.copy(),
        cell_type=mesh.cell_type,
        name=mesh.name + name_suffix,
        metadata={**mesh.metadata, "transform": "affine"},
    )


def transform_electrodes(electrodes: ElectrodeSet, transform: AffineTransform) -> ElectrodeSet:
    return ElectrodeSet(
        positions=transform.apply_points(electrodes.positions),
        labels=electrodes.labels,
        name=electrodes.name,
        metadata={**electrodes.metadata, "transform": "affine"},
    )


def transform_source_region(region: SourceRegion, transform: AffineTransform) -> SourceRegion:
    return SourceRegion(
        candidate_points=transform.apply_points(region.candidate_points),
        candidate_cell_ids=region.candidate_cell_ids.copy(),
        name=region.name,
        metadata={**region.metadata, "transform": "affine"},
    )


def transform_torso_geometry(
    geometry: TorsoGeometry, transform: AffineTransform, geometry_id: str | None = None
) -> TorsoGeometry:
    return TorsoGeometry(
        geometry_id=geometry_id or f"{geometry.geometry_id}_affine",
        volume_mesh=transform_mesh(geometry.volume_mesh, transform),
        surface_mesh=transform_mesh(geometry.surface_mesh, transform) if geometry.surface_mesh is not None else None,
        electrodes=transform_electrodes(geometry.electrodes, transform),
        source_region=transform_source_region(geometry.source_region, transform),
        registration_transform={"type": "affine", "matrix": transform.matrix, "offset": transform.offset},
        metadata={**geometry.metadata, "parent_geometry_id": geometry.geometry_id},
    )
