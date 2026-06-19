from __future__ import annotations

import numpy as np


def _as_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_equal_vectors(a, b, a_name: str = "a", b_name: str = "b") -> tuple[np.ndarray, np.ndarray]:
    first = _as_vector(a, a_name)
    second = _as_vector(b, b_name)
    if first.shape != second.shape:
        raise ValueError(f"{a_name} and {b_name} must have equal shape")
    return first, second


def l2_norm(x) -> float:
    """Return the Euclidean norm of a finite one-dimensional vector."""
    return float(np.linalg.norm(_as_vector(x, "x")))


def rmse(a, b) -> float:
    """Return root mean square error between equally shaped vectors."""
    first, second = _as_equal_vectors(a, b)
    return float(np.sqrt(np.mean((first - second) ** 2)))


def relative_l2_error(reference, estimate, eps: float = 1e-15) -> float:
    """Return ``||estimate-reference|| / max(||reference||, eps)``."""
    reference_array, estimate_array = _as_equal_vectors(reference, estimate, "reference", "estimate")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    denominator = max(float(np.linalg.norm(reference_array)), float(eps))
    return float(np.linalg.norm(estimate_array - reference_array) / denominator)


def max_abs_error(a, b) -> float:
    """Return maximum absolute error between equally shaped vectors."""
    first, second = _as_equal_vectors(a, b)
    return 0.0 if first.size == 0 else float(np.max(np.abs(first - second)))


def correlation(a, b, eps: float = 1e-15) -> float:
    """Return Pearson correlation, or NaN for an almost constant vector."""
    first, second = _as_equal_vectors(a, b)
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    first_centered = first - first.mean() if first.size else first
    second_centered = second - second.mean() if second.size else second
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator <= eps:
        return float("nan")
    return float(np.dot(first_centered, second_centered) / denominator)


def compute_snr_db(signal, noise, eps: float = 1e-15) -> float:
    """Return amplitude SNR in dB, with infinity for zero noise."""
    signal_array, noise_array = _as_equal_vectors(signal, noise, "signal", "noise")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    noise_norm = float(np.linalg.norm(noise_array))
    if noise_norm <= eps:
        return float("inf")
    signal_norm = float(np.linalg.norm(signal_array))
    return float(20.0 * np.log10(max(signal_norm, eps) / noise_norm))


def forward_signal_metrics(reference, estimate) -> dict[str, float]:
    """Return common errors and correlation for two forward signals."""
    return {
        "rmse": rmse(reference, estimate),
        "relative_l2_error": relative_l2_error(reference, estimate),
        "max_abs_error": max_abs_error(reference, estimate),
        "correlation": correlation(reference, estimate),
    }


def noise_metrics(clean, noisy, noise, eps: float = 1e-15) -> dict[str, float]:
    """Return signal/noise norms, achieved SNR and relative noise norm."""
    clean_array, noisy_array = _as_equal_vectors(clean, noisy, "clean", "noisy")
    _, noise_array = _as_equal_vectors(clean_array, noise, "clean", "noise")
    signal_norm = float(np.linalg.norm(clean_array))
    noise_norm = float(np.linalg.norm(noise_array))
    return {
        "signal_norm": signal_norm,
        "noisy_norm": float(np.linalg.norm(noisy_array)),
        "noise_norm": noise_norm,
        "snr_db": compute_snr_db(clean_array, noise_array, eps=eps),
        "relative_noise_norm": noise_norm / max(signal_norm, eps),
    }
