from __future__ import annotations

import numpy as np


def localization_error(estimated_position, true_position) -> float:
    """Return Euclidean source localization error."""
    return float(np.linalg.norm(np.asarray(estimated_position, dtype=float) - np.asarray(true_position, dtype=float)))


def moment_relative_error(estimated_moment, true_moment, eps: float = 1e-15) -> float:
    """Return relative L2 error of the dipole moment."""
    estimated = np.asarray(estimated_moment, dtype=float)
    true = np.asarray(true_moment, dtype=float)
    return float(np.linalg.norm(estimated - true) / max(float(np.linalg.norm(true)), float(eps)))


def moment_angle_error_deg(estimated_moment, true_moment, eps: float = 1e-15) -> float:
    """Return angular error between two moments in degrees, or NaN for zero moments."""
    estimated = np.asarray(estimated_moment, dtype=float)
    true = np.asarray(true_moment, dtype=float)
    denom = float(np.linalg.norm(estimated) * np.linalg.norm(true))
    if denom <= float(eps):
        return float("nan")
    cosine = float(np.dot(estimated, true) / denom)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(angle))


def is_successful_localization(error: float, threshold: float) -> bool:
    """Return whether localization error is within a threshold."""
    return bool(float(error) <= float(threshold))


def inverse_reconstruction_metrics(
    result,
    true_position,
    true_moment,
    localization_threshold: float | None = None,
) -> dict:
    """Return localization, moment and residual metrics for an inverse result."""
    loc_error = localization_error(result.estimated_position, true_position)
    metrics = {
        "localization_error": loc_error,
        "moment_relative_error": moment_relative_error(result.estimated_moment, true_moment),
        "moment_angle_error_deg": moment_angle_error_deg(result.estimated_moment, true_moment),
        "residual_norm": result.residual_norm,
        "relative_residual": result.relative_residual,
    }
    if localization_threshold is not None:
        metrics["success"] = is_successful_localization(loc_error, localization_threshold)
    return metrics
