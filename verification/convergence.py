from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConvergenceEntry:
    level: int
    h: float
    error: float


@dataclass(frozen=True)
class ConvergenceReport:
    entries: list[ConvergenceEntry]
    rates: list[float]

    @property
    def errors_decrease(self) -> bool:
        errors = np.asarray([entry.error for entry in self.entries], dtype=float)
        return bool(errors.size < 2 or np.all(np.diff(errors) < 0.0))

    @property
    def min_rate(self) -> float | None:
        return None if not self.rates else float(min(self.rates))


def _validate_series(h_values, errors) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(h_values, dtype=float)
    error = np.asarray(errors, dtype=float)
    if h.ndim != 1 or error.ndim != 1 or h.shape != error.shape:
        raise ValueError("h_values and errors must be one-dimensional arrays with equal shape")
    if h.size == 0:
        raise ValueError("at least one convergence entry is required")
    if np.any(~np.isfinite(h)) or np.any(~np.isfinite(error)):
        raise ValueError("h_values and errors must be finite")
    if np.any(h <= 0.0) or np.any(error <= 0.0):
        raise ValueError("h_values and errors must be positive")
    if h.size > 1 and np.any(np.diff(h) >= 0.0):
        raise ValueError("h_values must be strictly decreasing")
    return h, error


def estimate_rates(h_values, errors) -> list[float]:
    """Estimate pairwise rates ``log(e_i/e_j) / log(h_i/h_j)``."""
    h, error = _validate_series(h_values, errors)
    if h.size < 2:
        return []
    rates = np.log(error[:-1] / error[1:]) / np.log(h[:-1] / h[1:])
    return [float(rate) for rate in rates]


def build_convergence_report(h_values, errors) -> ConvergenceReport:
    """Build convergence entries and pairwise rates from refinement data."""
    h, error = _validate_series(h_values, errors)
    entries = [ConvergenceEntry(level=i, h=float(hi), error=float(ei)) for i, (hi, ei) in enumerate(zip(h, error))]
    return ConvergenceReport(entries=entries, rates=estimate_rates(h, error))


def format_convergence_report(report: ConvergenceReport) -> str:
    """Format a compact convergence table without printing it."""
    lines = ["level | h | error | rate", "----- | -------- | ------------ | --------"]
    for index, entry in enumerate(report.entries):
        rate = "-" if index == 0 else f"{report.rates[index - 1]:.4f}"
        lines.append(f"{entry.level:5d} | {entry.h:.6g} | {entry.error:.6e} | {rate}")
    return "\n".join(lines)
