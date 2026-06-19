from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sources import PointDipole

from .electrode_sets import ElectrodeSubset


@dataclass(frozen=True)
class ForwardBenchmarkScenario:
    """Cartesian forward benchmark configuration without large result arrays."""

    name: str
    geometry: Any
    sources: list[PointDipole]
    electrode_sets: list[ElectrodeSubset]
    noise_models: list[Any]
    reference: str = "average"
    reference_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", list(self.sources))
        object.__setattr__(self, "electrode_sets", list(self.electrode_sets))
        object.__setattr__(self, "noise_models", list(self.noise_models))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def validate(self) -> None:
        if not str(self.name).strip():
            raise ValueError("scenario name must not be empty")
        if not hasattr(self.geometry, "volume_mesh"):
            raise TypeError("scenario geometry must expose volume_mesh")
        if not self.sources:
            raise ValueError("scenario sources must not be empty")
        if not all(isinstance(source, PointDipole) for source in self.sources):
            raise TypeError("scenario sources must contain PointDipole objects")
        if not self.electrode_sets:
            raise ValueError("scenario electrode_sets must not be empty")
        if not all(isinstance(subset, ElectrodeSubset) for subset in self.electrode_sets):
            raise TypeError("scenario electrode_sets must contain ElectrodeSubset objects")
        if not self.noise_models:
            raise ValueError("scenario noise_models must not be empty")
        for model in self.noise_models:
            if not hasattr(model, "name") or not callable(getattr(model, "apply", None)):
                raise TypeError("each noise model must expose name and apply(values, rng=None)")
        if self.reference not in {"none", "average", "single"}:
            raise ValueError("reference must be one of: 'none', 'average', 'single'")
        if self.reference == "single" and self.reference_index is None:
            raise ValueError("reference_index is required for single reference")

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_sources": len(self.sources),
            "num_electrode_sets": len(self.electrode_sets),
            "num_noise_models": len(self.noise_models),
            "reference": self.reference,
            "reference_index": self.reference_index,
            "electrode_sets": [subset.to_config_dict() for subset in self.electrode_sets],
            "noise_models": [
                model.to_config_dict() if hasattr(model, "to_config_dict") else {"name": model.name}
                for model in self.noise_models
            ],
            "metadata": dict(self.metadata),
        }
