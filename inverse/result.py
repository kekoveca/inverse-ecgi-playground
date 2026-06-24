from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateInverseSolution:
    """Inverse LS solution for one candidate source position."""

    candidate_index: int
    position: np.ndarray
    cell_id: int | None
    moment: np.ndarray
    residual_norm: float
    relative_residual: float
    condition_number: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float)
        moment = np.asarray(self.moment, dtype=float)
        if position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {position.shape}")
        if moment.shape != (3,):
            raise ValueError(f"moment must have shape (3,), got {moment.shape}")
        object.__setattr__(self, "candidate_index", int(self.candidate_index))
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "cell_id", None if self.cell_id is None else int(self.cell_id))
        object.__setattr__(self, "moment", moment)
        object.__setattr__(self, "residual_norm", float(self.residual_norm))
        object.__setattr__(self, "relative_residual", float(self.relative_residual))
        if self.condition_number is not None:
            object.__setattr__(self, "condition_number", float(self.condition_number))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_row(self) -> dict[str, Any]:
        """Return a CSV-friendly row without vector-valued fields."""
        return {
            "candidate_index": self.candidate_index,
            "x": float(self.position[0]),
            "y": float(self.position[1]),
            "z": float(self.position[2]),
            "cell_id": self.cell_id,
            "px": float(self.moment[0]),
            "py": float(self.moment[1]),
            "pz": float(self.moment[2]),
            "residual_norm": self.residual_norm,
            "relative_residual": self.relative_residual,
            "condition_number": self.condition_number,
        }


@dataclass(frozen=True)
class SingleDipoleInverseResult:
    """Result of exhaustive single-dipole search over candidate points."""

    observed_measurements: np.ndarray
    best: CandidateInverseSolution
    candidates: list[CandidateInverseSolution]
    lambda_reg: float
    reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observed = np.asarray(self.observed_measurements, dtype=float)
        if observed.ndim != 1:
            raise ValueError("observed_measurements must be one-dimensional")
        candidates = list(self.candidates)
        if not candidates:
            raise ValueError("candidates must contain at least one solution")
        object.__setattr__(self, "observed_measurements", observed)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "lambda_reg", float(self.lambda_reg))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def estimated_position(self) -> np.ndarray:
        return self.best.position

    @property
    def estimated_cell_id(self) -> int | None:
        return self.best.cell_id

    @property
    def estimated_moment(self) -> np.ndarray:
        return self.best.moment

    @property
    def best_candidate_index(self) -> int:
        return self.best.candidate_index

    @property
    def residual_norm(self) -> float:
        return self.best.residual_norm

    @property
    def relative_residual(self) -> float:
        return self.best.relative_residual

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)

    def residual_map(self) -> np.ndarray:
        """Return candidate residual norms in solved-candidate order."""
        return np.asarray([candidate.residual_norm for candidate in self.candidates], dtype=float)

    def moment_map(self) -> np.ndarray:
        """Return estimated moments in solved-candidate order."""
        return np.asarray([candidate.moment for candidate in self.candidates], dtype=float)

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a compact summary without full residual/moment maps."""
        return {
            "best_candidate_index": self.best_candidate_index,
            "estimated_position": self.estimated_position.tolist(),
            "estimated_cell_id": self.estimated_cell_id,
            "estimated_moment": self.estimated_moment.tolist(),
            "residual_norm": self.residual_norm,
            "relative_residual": self.relative_residual,
            "num_candidates": self.num_candidates,
            "lambda_reg": self.lambda_reg,
            "reference": self.reference,
            "metadata": dict(self.metadata),
        }
