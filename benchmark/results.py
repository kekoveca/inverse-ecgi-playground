from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _measurement_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


@dataclass(frozen=True)
class ForwardBenchmarkRecord:
    scenario_name: str
    source_index: int
    source_position: np.ndarray
    source_moment: np.ndarray
    source_cell_id: int | None
    electrode_set_name: str
    num_electrodes: int
    noise_model_name: str
    reference: str
    clean_measurements: np.ndarray
    noisy_measurements: np.ndarray
    noise: np.ndarray
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        position = np.asarray(self.source_position, dtype=float)
        moment = np.asarray(self.source_moment, dtype=float)
        if position.shape != (3,) or moment.shape != (3,):
            raise ValueError("source_position and source_moment must have shape (3,)")
        clean = _measurement_vector(self.clean_measurements, "clean_measurements")
        noisy = _measurement_vector(self.noisy_measurements, "noisy_measurements")
        noise = _measurement_vector(self.noise, "noise")
        if clean.shape != noisy.shape or clean.shape != noise.shape:
            raise ValueError("clean_measurements, noisy_measurements and noise must have equal shape")
        if int(self.num_electrodes) != clean.size:
            raise ValueError("num_electrodes must match measurement vector length")
        object.__setattr__(self, "source_position", position.copy())
        object.__setattr__(self, "source_moment", moment.copy())
        object.__setattr__(self, "clean_measurements", clean)
        object.__setattr__(self, "noisy_measurements", noisy)
        object.__setattr__(self, "noise", noise)
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_row(self) -> dict[str, Any]:
        row = {
            "scenario_name": self.scenario_name,
            "source_index": int(self.source_index),
            "x": float(self.source_position[0]),
            "y": float(self.source_position[1]),
            "z": float(self.source_position[2]),
            "px": float(self.source_moment[0]),
            "py": float(self.source_moment[1]),
            "pz": float(self.source_moment[2]),
            "source_cell_id": self.source_cell_id,
            "electrode_set_name": self.electrode_set_name,
            "num_electrodes": int(self.num_electrodes),
            "noise_model_name": self.noise_model_name,
            "reference": self.reference,
            "clean_norm": float(np.linalg.norm(self.clean_measurements)),
            "noisy_norm": float(np.linalg.norm(self.noisy_measurements)),
            "noise_norm": float(np.linalg.norm(self.noise)),
            "snr_db": self.metrics.get("snr_db", float("inf") if np.linalg.norm(self.noise) == 0.0 else None),
        }
        row.update(self.metrics)
        return row


@dataclass(frozen=True)
class ForwardBenchmarkResult:
    scenario: Any
    records: list[ForwardBenchmarkRecord]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(isinstance(record, ForwardBenchmarkRecord) for record in self.records):
            raise TypeError("records must contain ForwardBenchmarkRecord objects")
        object.__setattr__(self, "records", list(self.records))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def __len__(self) -> int:
        return len(self.records)

    def to_rows(self) -> list[dict[str, Any]]:
        return [record.to_row() for record in self.records]

    def summary(self) -> dict[str, Any]:
        scenario_name = getattr(self.scenario, "name", None)
        if scenario_name is None and isinstance(self.scenario, dict):
            scenario_name = self.scenario.get("name")
        scenario_sources = getattr(self.scenario, "sources", None)
        if scenario_sources is not None:
            num_sources = len(scenario_sources)
        elif isinstance(self.scenario, dict) and "num_sources" in self.scenario:
            num_sources = int(self.scenario["num_sources"])
        else:
            num_sources = len({record.source_index for record in self.records})

        electrode_names = list(dict.fromkeys(record.electrode_set_name for record in self.records))
        noise_names = list(dict.fromkeys(record.noise_model_name for record in self.records))
        return {
            "num_records": len(self.records),
            "scenario_name": scenario_name,
            "num_sources": num_sources,
            "electrode_set_names": electrode_names,
            "noise_model_names": noise_names,
            "metadata": dict(self.metadata),
        }
