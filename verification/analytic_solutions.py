from __future__ import annotations

import numpy as np


def homogeneous_free_space_dipole_potential(
    points,
    position,
    moment,
    conductivity: float = 1.0,
    singular_value: float = np.nan,
) -> np.ndarray:
    """Evaluate the infinite homogeneous-space dipole potential.

    This diagnostic reference ignores torso boundaries. Values exactly at the
    source are replaced by ``singular_value``.
    """
    points = np.asarray(points, dtype=float)
    position = np.asarray(position, dtype=float)
    moment = np.asarray(moment, dtype=float)
    if points.shape[-1] != 3:
        raise ValueError("points must have shape (..., 3)")
    if position.shape != (3,) or moment.shape != (3,):
        raise ValueError("position and moment must have shape (3,)")
    if conductivity <= 0.0:
        raise ValueError("conductivity must be positive")

    displacement = points - position
    radius = np.linalg.norm(displacement, axis=-1)
    numerator = np.einsum("...i,i->...", displacement, moment)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = numerator / (4.0 * np.pi * conductivity * radius**3)
    return np.where(radius == 0.0, singular_value, values)
