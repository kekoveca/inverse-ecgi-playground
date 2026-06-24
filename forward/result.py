from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sources import PointDipole


@dataclass(frozen=True)
class ForwardResult:
    """Result of one point-dipole forward solve.

    It keeps the DOLFINx potential, a copied nodal array, raw/referenced
    electrode values and lightweight metadata suitable for summaries.
    ``nodal_values`` is retained for backward compatibility and is ordered by
    ``nodal_value_ordering``; forward solves currently store DOLFINx dof order.
    """

    source: PointDipole
    potential: Any
    nodal_values: np.ndarray
    raw_measurements: np.ndarray
    measurements: np.ndarray
    reference: str
    metadata: dict[str, Any] = field(default_factory=dict)
    nodal_value_ordering: str = "dolfinx_dof"
    meshdata_nodal_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        nodal_values = np.asarray(self.nodal_values, dtype=float)
        raw_measurements = np.asarray(self.raw_measurements, dtype=float)
        measurements = np.asarray(self.measurements, dtype=float)
        nodal_value_ordering = str(self.nodal_value_ordering)
        if nodal_values.ndim != 1:
            raise ValueError("nodal_values must be one-dimensional")
        if raw_measurements.ndim != 1:
            raise ValueError("raw_measurements must be one-dimensional")
        if measurements.ndim != 1:
            raise ValueError("measurements must be one-dimensional")
        if raw_measurements.shape != measurements.shape:
            raise ValueError("raw_measurements and measurements must have the same shape")
        if nodal_value_ordering not in {"dolfinx_dof", "meshdata_node", "unknown"}:
            raise ValueError("nodal_value_ordering must be 'dolfinx_dof', 'meshdata_node', or 'unknown'")
        meshdata_nodal_values = self.meshdata_nodal_values
        if meshdata_nodal_values is not None:
            meshdata_nodal_values = np.asarray(meshdata_nodal_values, dtype=float)
            if meshdata_nodal_values.ndim != 1:
                raise ValueError("meshdata_nodal_values must be one-dimensional")
        object.__setattr__(self, "nodal_values", nodal_values)
        object.__setattr__(self, "raw_measurements", raw_measurements)
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "nodal_value_ordering", nodal_value_ordering)
        object.__setattr__(self, "meshdata_nodal_values", meshdata_nodal_values)

    @property
    def dof_values(self) -> np.ndarray:
        """Return the copied DOLFINx dof-ordered potential values."""
        if self.nodal_value_ordering != "dolfinx_dof":
            raise ValueError(f"nodal_values are ordered as {self.nodal_value_ordering!r}, not DOLFINx dofs")
        return self.nodal_values

    @property
    def has_meshdata_nodal_values(self) -> bool:
        return self.meshdata_nodal_values is not None

    @property
    def num_nodes(self) -> int:
        return int(self.nodal_values.shape[0])

    @property
    def num_electrodes(self) -> int:
        return int(self.measurements.shape[0])

    @property
    def measurement_norm(self) -> float:
        return float(np.linalg.norm(self.measurements))

    @property
    def raw_measurement_norm(self) -> float:
        return float(np.linalg.norm(self.raw_measurements))

    def to_dict(self) -> dict[str, Any]:
        """Return a metadata-friendly summary without large arrays."""
        return {
            "source_name": self.source.name,
            "source_position": self.source.position.tolist(),
            "source_moment": self.source.moment.tolist(),
            "source_cell_id": self.source.cell_id,
            "num_nodes": self.num_nodes,
            "num_electrodes": self.num_electrodes,
            "nodal_value_ordering": self.nodal_value_ordering,
            "has_meshdata_nodal_values": self.has_meshdata_nodal_values,
            "measurement_norm": self.measurement_norm,
            "raw_measurement_norm": self.raw_measurement_norm,
            "reference": self.reference,
            "metadata": dict(self.metadata),
        }
