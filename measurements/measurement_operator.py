from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geometry import ElectrodeSet, MeshData

from .electrode_locator import locate_electrodes_in_mesh
from .interpolation import build_point_interpolation_matrix
from .reference import reference_matrix


@dataclass(frozen=True)
class MeasurementOperator:
    interpolation_matrix: Any
    reference_matrix: Any
    electrode_cell_ids: np.ndarray
    electrode_barycentric: np.ndarray
    reference: str
    reference_index: int | None = None
    labels: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cell_ids = np.asarray(self.electrode_cell_ids, dtype=np.int64)
        barycentric = np.asarray(self.electrode_barycentric, dtype=float)
        if cell_ids.ndim != 1:
            raise ValueError("electrode_cell_ids must be one-dimensional")
        if barycentric.shape != (cell_ids.shape[0], 4):
            raise ValueError(f"electrode_barycentric must have shape ({cell_ids.shape[0]}, 4)")
        if self.labels is not None and len(self.labels) != cell_ids.shape[0]:
            raise ValueError("labels length must match number of electrodes")
        object.__setattr__(self, "electrode_cell_ids", cell_ids)
        object.__setattr__(self, "electrode_barycentric", barycentric)
        if self.labels is not None:
            object.__setattr__(self, "labels", list(self.labels))

    @property
    def num_electrodes(self) -> int:
        return int(self.interpolation_matrix.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.interpolation_matrix.shape[1])

    def raw_matrix(self):
        return self.interpolation_matrix

    def matrix(self):
        return self.reference_matrix @ self.interpolation_matrix

    def evaluate_raw(self, nodal_values) -> np.ndarray:
        values = np.asarray(nodal_values, dtype=float)
        if values.shape != (self.num_nodes,):
            raise ValueError(f"nodal_values must have shape ({self.num_nodes},), got {values.shape}")
        return np.asarray(self.interpolation_matrix @ values, dtype=float)

    def evaluate(self, nodal_values) -> np.ndarray:
        return np.asarray(self.reference_matrix @ self.evaluate_raw(nodal_values), dtype=float)


def build_measurement_operator(
    mesh: MeshData,
    electrodes: ElectrodeSet,
    reference: str = "average",
    reference_index: int | None = None,
    sparse: bool = True,
    tol: float = 1e-10,
) -> MeasurementOperator:
    """Build the electrode measurement operator ``M = R @ P``."""
    cell_ids, barycentric = locate_electrodes_in_mesh(mesh, electrodes, tol=tol)
    interpolation_matrix = build_point_interpolation_matrix(
        mesh,
        electrodes.positions,
        cell_ids=cell_ids,
        barycentric=barycentric,
        sparse=sparse,
        tol=tol,
    )
    ref_matrix = reference_matrix(
        electrodes.num_electrodes,
        reference=reference,
        reference_index=reference_index,
        sparse=sparse,
    )
    return MeasurementOperator(
        interpolation_matrix=interpolation_matrix,
        reference_matrix=ref_matrix,
        electrode_cell_ids=cell_ids,
        electrode_barycentric=barycentric,
        reference=reference,
        reference_index=reference_index,
        labels=electrodes.labels,
        metadata={"mesh_name": mesh.name, "electrode_set": electrodes.name},
    )
