from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from scipy.spatial import cKDTree

import numpy as np

from .mesh_model import MeshData


@dataclass(frozen=True)
class ElectrodeSet:
    positions: np.ndarray
    labels: list[str] | None = None
    name: str = "electrodes"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2:
            raise ValueError("positions must have shape (n_electrodes, geometric_dim)")
        if positions.shape[1] not in (2, 3):
            raise ValueError("electrode positions must be 2D or 3D coordinates")

        labels = self.labels
        if labels is None:
            labels = [f"E{i:03d}" for i in range(positions.shape[0])]
        if len(labels) != positions.shape[0]:
            raise ValueError("labels length must match number of electrodes")

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "labels", list(labels))

    @property
    def num_electrodes(self) -> int:
        return int(self.positions.shape[0])

    @property
    def geometric_dim(self) -> int:
        return int(self.positions.shape[1])

    def centered(self) -> "ElectrodeSet":
        return ElectrodeSet(
            positions=self.positions - self.positions.mean(axis=0, keepdims=True),
            labels=self.labels,
            name=self.name,
            metadata=self.metadata,
        )

    def nearest_mesh_nodes(self, mesh: MeshData) -> np.ndarray:
        """Return nearest mesh node index for every electrode"""
        if mesh.geometric_dim != self.geometric_dim:
            raise ValueError("mesh and electrodes must have the same geometric dimension")

        tree = cKDTree(mesh.points)
        _, indices = tree.query(self.positions, k=1)
        return indices.astype(np.int64)

    def distance_to_mesh_nodes(self, mesh: MeshData) -> np.ndarray:
        ids = self.nearest_mesh_nodes(mesh)
        return np.linalg.norm(self.positions - mesh.points[ids], axis=1)


@dataclass(frozen=True)
class ElectrodePlacementReport:
    mean_distance_to_nearest_node: float
    max_distance_to_nearest_node: float
    nearest_node_ids: np.ndarray


def electrode_placement_report(electrodes: ElectrodeSet, mesh: MeshData) -> ElectrodePlacementReport:
    ids = electrodes.nearest_mesh_nodes(mesh)
    distances = np.linalg.norm(electrodes.positions - mesh.points[ids], axis=1)
    return ElectrodePlacementReport(
        mean_distance_to_nearest_node=float(distances.mean()),
        max_distance_to_nearest_node=float(distances.max()),
        nearest_node_ids=ids,
    )
