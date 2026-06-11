from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PointDipole:
    position: np.ndarray
    moment: np.ndarray
    cell_id: int | None = None
    name: str = "point_dipole"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float)
        moment = np.asarray(self.moment, dtype=float)

        if position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {position.shape}")
        if moment.shape != (3,):
            raise ValueError(f"moment must have shape (3,), got {moment.shape}")
        if not np.all(np.isfinite(position)):
            raise ValueError("position must contain only finite values")
        if not np.all(np.isfinite(moment)):
            raise ValueError("moment must contain only finite values")
        if self.cell_id is not None and int(self.cell_id) < 0:
            raise ValueError("cell_id must be non-negative")

        object.__setattr__(self, "position", position)
        object.__setattr__(self, "moment", moment)
        if self.cell_id is not None:
            object.__setattr__(self, "cell_id", int(self.cell_id))

    def with_cell_id(self, cell_id: int) -> "PointDipole":
        return replace(self, cell_id=int(cell_id))

    def normalized_moment(self) -> np.ndarray:
        norm = float(np.linalg.norm(self.moment))
        if norm == 0.0:
            raise ValueError("cannot normalize zero dipole moment")
        return self.moment / norm
