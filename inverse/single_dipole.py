from __future__ import annotations

import numpy as np

from .regularization import condition_number, relative_residual, residual_norm, solve_tikhonov_moment
from .result import CandidateInverseSolution, SingleDipoleInverseResult


class SingleDipoleInverseSolver:
    """Exhaustive single-dipole inverse over a GreenTransferMatrix grid."""

    def __init__(
        self,
        transfer_matrix,
        lambda_reg: float = 0.0,
        reference: str | None = None,
        eps: float = 1e-15,
    ) -> None:
        self.transfer_matrix = transfer_matrix
        self.lambda_reg = float(lambda_reg)
        if self.lambda_reg < 0.0:
            raise ValueError("lambda_reg must be non-negative")
        self.reference = reference
        self.eps = float(eps)
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

    def _validate_measurements(self, measurements) -> np.ndarray:
        values = np.asarray(measurements, dtype=float)
        expected = int(self.transfer_matrix.num_measurements)
        if values.shape != (expected,):
            raise ValueError(f"measurements must have shape ({expected},), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("measurements must contain only finite values")
        return values

    def solve_candidate(self, candidate_index: int, measurements) -> CandidateInverseSolution:
        """Solve the regularized moment problem for one candidate."""
        g = self._validate_measurements(measurements)
        candidate_index = int(candidate_index)
        if candidate_index < 0 or candidate_index >= self.transfer_matrix.num_candidates:
            raise IndexError("candidate_index is outside the transfer matrix")
        A = self.transfer_matrix.matrix_for_candidate(candidate_index)
        moment = solve_tikhonov_moment(A, g, lambda_reg=self.lambda_reg)
        return CandidateInverseSolution(
            candidate_index=candidate_index,
            position=self.transfer_matrix.candidate_points[candidate_index],
            cell_id=int(self.transfer_matrix.candidate_cell_ids[candidate_index]),
            moment=moment,
            residual_norm=residual_norm(A, moment, g),
            relative_residual=relative_residual(A, moment, g, eps=self.eps),
            condition_number=condition_number(A),
            metadata={"lambda_reg": self.lambda_reg},
        )

    def solve(self, measurements, candidate_indices=None) -> SingleDipoleInverseResult:
        """Solve all selected candidates and return the minimum-residual result."""
        g = self._validate_measurements(measurements)
        if candidate_indices is None:
            indices = np.arange(self.transfer_matrix.num_candidates, dtype=np.int64)
        else:
            indices = np.asarray(list(candidate_indices), dtype=np.int64)
            if indices.ndim != 1:
                raise ValueError("candidate_indices must be one-dimensional")
            if indices.size == 0:
                raise ValueError("candidate_indices must contain at least one index")
            if np.any(indices < 0) or np.any(indices >= self.transfer_matrix.num_candidates):
                raise IndexError("candidate_indices contain an index outside the transfer matrix")

        candidates = [self.solve_candidate(int(index), g) for index in indices]
        best = min(candidates, key=lambda candidate: candidate.residual_norm)
        return SingleDipoleInverseResult(
            observed_measurements=g,
            best=best,
            candidates=candidates,
            lambda_reg=self.lambda_reg,
            reference=self.reference,
            metadata={
                "num_transfer_candidates": self.transfer_matrix.num_candidates,
                "candidate_indices": indices.tolist(),
            },
        )


def solve_single_dipole_inverse(
    transfer_matrix,
    measurements,
    lambda_reg: float = 0.0,
    candidate_indices=None,
    reference: str | None = None,
) -> SingleDipoleInverseResult:
    """Convenience wrapper for single-dipole inverse reconstruction."""
    return SingleDipoleInverseSolver(
        transfer_matrix=transfer_matrix,
        lambda_reg=lambda_reg,
        reference=reference,
    ).solve(measurements, candidate_indices=candidate_indices)
