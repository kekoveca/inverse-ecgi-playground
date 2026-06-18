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
    """

    source: PointDipole
    potential: Any
    nodal_values: np.ndarray
    raw_measurements: np.ndarray
    measurements: np.ndarray
    reference: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nodal_values = np.asarray(self.nodal_values, dtype=float)
        raw_measurements = np.asarray(self.raw_measurements, dtype=float)
        measurements = np.asarray(self.measurements, dtype=float)
        if nodal_values.ndim != 1:
            raise ValueError("nodal_values must be one-dimensional")
        if raw_measurements.ndim != 1:
            raise ValueError("raw_measurements must be one-dimensional")
        if measurements.ndim != 1:
            raise ValueError("measurements must be one-dimensional")
        if raw_measurements.shape != measurements.shape:
            raise ValueError("raw_measurements and measurements must have the same shape")
        object.__setattr__(self, "nodal_values", nodal_values)
        object.__setattr__(self, "raw_measurements", raw_measurements)
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "metadata", dict(self.metadata))

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
            "measurement_norm": self.measurement_norm,
            "raw_measurement_norm": self.raw_measurement_norm,
            "reference": self.reference,
            "metadata": dict(self.metadata),
        }
