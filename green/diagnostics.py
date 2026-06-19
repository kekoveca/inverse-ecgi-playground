from __future__ import annotations

import numpy as np


def compare_forward_and_green(
    forward_result,
    transfer_matrix,
    candidate_index: int,
    moment,
    allow_sign_flip: bool = True,
    eps: float = 1e-15,
) -> dict:
    """Compare ordinary FEM measurements with both Green sign conventions."""
    forward = np.asarray(forward_result.measurements, dtype=float)
    moment = np.asarray(moment, dtype=float)
    if moment.shape != (3,):
        raise ValueError(f"moment must have shape (3,), got {moment.shape}")
    green_plus = np.asarray(transfer_matrix.A[int(candidate_index)] @ moment, dtype=float)
    if green_plus.shape != forward.shape:
        raise ValueError(
            f"Green prediction shape {green_plus.shape} does not match forward measurements {forward.shape}"
        )
    green_minus = -green_plus
    scale = max(float(np.linalg.norm(forward)), float(eps))
    error_plus = float(np.linalg.norm(green_plus - forward) / scale)
    error_minus = float(np.linalg.norm(green_minus - forward) / scale)
    best_sign = 1.0
    best_error = error_plus
    if allow_sign_flip and error_minus < error_plus:
        best_sign = -1.0
        best_error = error_minus
    return {
        "rel_error_plus": error_plus,
        "rel_error_minus": error_minus,
        "best_sign": best_sign,
        "best_rel_error": best_error,
        "forward_norm": float(np.linalg.norm(forward)),
        "green_norm_plus": float(np.linalg.norm(green_plus)),
        "green_norm_minus": float(np.linalg.norm(green_minus)),
    }


def infer_green_sign_from_cases(cases) -> float:
    """Infer the majority best sign from consistency diagnostic dictionaries."""
    cases = list(cases)
    if not cases:
        raise ValueError("cases must contain at least one consistency diagnostic")
    signs = np.asarray([float(case["best_sign"]) for case in cases], dtype=float)
    return 1.0 if np.count_nonzero(signs > 0) >= np.count_nonzero(signs < 0) else -1.0
