from __future__ import annotations

import numpy as np


def _validate_system(A, g) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    g = np.asarray(g, dtype=float)
    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError(f"A must have shape (num_measurements, 3), got {A.shape}")
    if g.shape != (A.shape[0],):
        raise ValueError(f"g must have shape ({A.shape[0]},), got {g.shape}")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(g)):
        raise ValueError("A and g must contain only finite values")
    return A, g


def solve_tikhonov_moment(A, g, lambda_reg: float = 0.0, rcond=None) -> np.ndarray:
    """Solve the 3-component dipole moment LS/Tikhonov problem.

    For ``lambda_reg == 0`` this uses ``np.linalg.lstsq`` for robust handling
    of rank-deficient candidate matrices. Positive regularization solves
    ``(A.T @ A + lambda I) p = A.T @ g``.
    """
    A, g = _validate_system(A, g)
    lambda_reg = float(lambda_reg)
    if lambda_reg < 0.0:
        raise ValueError("lambda_reg must be non-negative")
    if lambda_reg == 0.0:
        moment, *_ = np.linalg.lstsq(A, g, rcond=rcond)
        return np.asarray(moment, dtype=float)
    lhs = A.T @ A + lambda_reg * np.eye(3)
    rhs = A.T @ g
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=rcond)[0]


def residual_vector(A, p, g) -> np.ndarray:
    """Return ``A @ p - g`` for a candidate solution."""
    A, g = _validate_system(A, g)
    p = np.asarray(p, dtype=float)
    if p.shape != (3,):
        raise ValueError(f"p must have shape (3,), got {p.shape}")
    return np.asarray(A @ p - g, dtype=float)


def residual_norm(A, p, g) -> float:
    """Return the Euclidean residual norm."""
    return float(np.linalg.norm(residual_vector(A, p, g)))


def relative_residual(A, p, g, eps: float = 1e-15) -> float:
    """Return residual norm normalized by the observed measurement norm."""
    g = np.asarray(g, dtype=float)
    return float(residual_norm(A, p, g) / max(float(np.linalg.norm(g)), float(eps)))


def condition_number(A) -> float:
    """Return ``cond(A)`` or infinity if it cannot be estimated."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be two-dimensional")
    try:
        value = float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        return float("inf")
    return value if np.isfinite(value) else float("inf")
