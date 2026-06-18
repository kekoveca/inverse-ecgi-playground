from __future__ import annotations

import numpy as np


def _as_points(x) -> np.ndarray:
    points = np.asarray(x, dtype=float)
    if points.shape[-1] != 3:
        raise ValueError("x must have shape (..., 3)")
    return points


def u_exact_neumann_cosine(x) -> np.ndarray:
    """Return ``cos(2 pi x) cos(2 pi y) cos(2 pi z)`` on the unit cube."""
    points = _as_points(x)
    return np.prod(np.cos(2.0 * np.pi * points), axis=-1)


def rhs_neumann_cosine(x) -> np.ndarray:
    """Return ``-Delta u = 12 pi^2 u`` for the cosine solution."""
    return 12.0 * np.pi**2 * u_exact_neumann_cosine(x)
