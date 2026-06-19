from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .gradients import gradients_at_candidate_cells, locate_candidate_points_in_dolfinx


@dataclass
class GreenTransferMatrix:
    """Dipole transfer tensor ``A[j, i, :] = grad G_i(x_j)``."""

    A: np.ndarray
    candidate_points: np.ndarray
    candidate_cell_ids: np.ndarray
    sign: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.candidate_points = np.asarray(self.candidate_points, dtype=float)
        self.candidate_cell_ids = np.asarray(self.candidate_cell_ids, dtype=np.int64)
        self.sign = float(self.sign)
        if self.A.ndim != 3 or self.A.shape[2] != 3:
            raise ValueError("A must have shape (num_candidates, num_measurements, 3)")
        if self.candidate_points.shape != (self.A.shape[0], 3):
            raise ValueError("candidate_points must have shape (num_candidates, 3)")
        if self.candidate_cell_ids.shape != (self.A.shape[0],):
            raise ValueError("candidate_cell_ids must have shape (num_candidates,)")
        if not np.isfinite(self.sign) or self.sign == 0.0:
            raise ValueError("sign must be a finite nonzero number")

    @property
    def num_candidates(self) -> int:
        return int(self.A.shape[0])

    @property
    def num_measurements(self) -> int:
        return int(self.A.shape[1])

    def matrix_for_candidate(self, candidate_index: int) -> np.ndarray:
        """Return the signed ``(num_measurements, 3)`` matrix for a candidate."""
        return self.sign * self.A[int(candidate_index)]

    def predict(self, candidate_index: int, moment) -> np.ndarray:
        """Predict referenced measurements for one candidate and dipole moment."""
        moment = np.asarray(moment, dtype=float)
        if moment.shape != (3,):
            raise ValueError(f"moment must have shape (3,), got {moment.shape}")
        return np.asarray(self.matrix_for_candidate(candidate_index) @ moment, dtype=float)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "num_candidates": self.num_candidates,
            "num_measurements": self.num_measurements,
            "sign": self.sign,
            "metadata": self.metadata,
        }


def build_green_transfer_matrix(
    poisson_solver,
    green_basis,
    candidate_points,
    candidate_cell_ids=None,
    tol: float = 1e-10,
) -> GreenTransferMatrix:
    """Build the Green transfer tensor at candidate source points."""
    if not green_basis.has_functions:
        raise ValueError("GreenBasis has no retained functions; use GreenSolver(keep_functions=True)")
    points = np.asarray(candidate_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"candidate_points must have shape (n, 3), got {points.shape}")
    if candidate_cell_ids is None:
        cell_ids = locate_candidate_points_in_dolfinx(poisson_solver, points, tol=tol)
    else:
        cell_ids = np.asarray(candidate_cell_ids, dtype=np.int64)
        if cell_ids.shape != (points.shape[0],):
            raise ValueError("candidate_cell_ids must have one DOLFINx cell id per candidate point")

    gradients = gradients_at_candidate_cells(poisson_solver, green_basis.functions, cell_ids)
    A = np.transpose(gradients, (1, 0, 2))
    return GreenTransferMatrix(
        A=A,
        candidate_points=points,
        candidate_cell_ids=cell_ids,
        sign=green_basis.sign,
        metadata={
            "reference": green_basis.reference,
            "row_indices": green_basis.metadata.get("row_indices"),
            "sign_convention": green_basis.metadata.get("sign_convention"),
        },
    )
