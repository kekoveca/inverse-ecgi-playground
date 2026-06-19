from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import compute_snr_db


class NoiseModel:
    """Interface for reproducible additive benchmark noise models."""

    name: str

    def apply(self, values: np.ndarray, rng=None) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def to_config_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.__class__.__name__}


def _values_copy(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must contain only finite values")
    return array.copy()


@dataclass(frozen=True)
class NoNoise(NoiseModel):
    name: str = "none"

    def apply(self, values: np.ndarray, rng=None) -> tuple[np.ndarray, np.ndarray]:
        clean = _values_copy(values)
        return clean, np.zeros_like(clean)


@dataclass(frozen=True)
class AbsoluteGaussianNoise(NoiseModel):
    sigma: float
    seed: int | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma) or self.sigma < 0.0:
            raise ValueError("sigma must be finite and non-negative")
        if self.name is None:
            object.__setattr__(self, "name", f"gaussian_sigma_{self.sigma:g}")

    def apply(self, values: np.ndarray, rng=None) -> tuple[np.ndarray, np.ndarray]:
        clean = _values_copy(values)
        generator = np.random.default_rng(self.seed) if rng is None else rng
        noise = generator.normal(0.0, self.sigma, size=clean.shape)
        return clean + noise, noise

    def to_config_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.__class__.__name__, "sigma": self.sigma, "seed": self.seed}


@dataclass(frozen=True)
class RelativeGaussianNoise(NoiseModel):
    snr_db: float
    seed: int | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.snr_db):
            raise ValueError("snr_db must be finite")
        if self.name is None:
            object.__setattr__(self, "name", f"gaussian_snr_{self.snr_db:g}db")

    def apply(self, values: np.ndarray, rng=None) -> tuple[np.ndarray, np.ndarray]:
        clean = _values_copy(values)
        signal_norm = float(np.linalg.norm(clean))
        if signal_norm <= 1e-15 or clean.size == 0:
            return clean, np.zeros_like(clean)

        generator = np.random.default_rng(self.seed) if rng is None else rng
        direction = generator.normal(size=clean.shape)
        direction_norm = float(np.linalg.norm(direction))
        while direction_norm <= 1e-15:
            direction = generator.normal(size=clean.shape)
            direction_norm = float(np.linalg.norm(direction))
        target_norm = signal_norm / (10.0 ** (self.snr_db / 20.0))
        noise = direction * (target_norm / direction_norm)
        return clean + noise, noise

    def to_config_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.__class__.__name__, "snr_db": self.snr_db, "seed": self.seed}


__all__ = [
    "AbsoluteGaussianNoise",
    "NoNoise",
    "NoiseModel",
    "RelativeGaussianNoise",
    "compute_snr_db",
]
