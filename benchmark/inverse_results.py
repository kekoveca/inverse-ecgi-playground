from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InverseBenchmarkRecord:
    scenario_name: str
    source_index: int
    source_position: np.ndarray
    source_moment: np.ndarray
    source_cell_id: int | None
    electrode_set_name: str
    num_electrodes: int
    noise_model_name: str
    measurement_kind: str
    lambda_reg: float
    estimated_candidate_index: int
    estimated_position: np.ndarray
    estimated_cell_id: int | None
    estimated_moment: np.ndarray
    residual_norm: float
    relative_residual: float
    localization_error: float
    moment_relative_error: float
    moment_angle_error_deg: float
    success: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_position = np.asarray(self.source_position, dtype=float)
        source_moment = np.asarray(self.source_moment, dtype=float)
        estimated_position = np.asarray(self.estimated_position, dtype=float)
        estimated_moment = np.asarray(self.estimated_moment, dtype=float)
        for name, value in (
            ("source_position", source_position),
            ("source_moment", source_moment),
            ("estimated_position", estimated_position),
            ("estimated_moment", estimated_moment),
        ):
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,)")
        if self.measurement_kind not in {"clean", "noisy"}:
            raise ValueError("measurement_kind must be 'clean' or 'noisy'")
        object.__setattr__(self, "source_position", source_position)
        object.__setattr__(self, "source_moment", source_moment)
        object.__setattr__(self, "estimated_position", estimated_position)
        object.__setattr__(self, "estimated_moment", estimated_moment)
        object.__setattr__(self, "lambda_reg", float(self.lambda_reg))
        object.__setattr__(self, "residual_norm", float(self.residual_norm))
        object.__setattr__(self, "relative_residual", float(self.relative_residual))
        object.__setattr__(self, "localization_error", float(self.localization_error))
        object.__setattr__(self, "moment_relative_error", float(self.moment_relative_error))
        object.__setattr__(self, "moment_angle_error_deg", float(self.moment_angle_error_deg))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_row(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "source_index": int(self.source_index),
            "true_x": float(self.source_position[0]),
            "true_y": float(self.source_position[1]),
            "true_z": float(self.source_position[2]),
            "true_px": float(self.source_moment[0]),
            "true_py": float(self.source_moment[1]),
            "true_pz": float(self.source_moment[2]),
            "source_cell_id": self.source_cell_id,
            "electrode_set_name": self.electrode_set_name,
            "num_electrodes": int(self.num_electrodes),
            "noise_model_name": self.noise_model_name,
            "measurement_kind": self.measurement_kind,
            "lambda_reg": self.lambda_reg,
            "estimated_candidate_index": int(self.estimated_candidate_index),
            "estimated_x": float(self.estimated_position[0]),
            "estimated_y": float(self.estimated_position[1]),
            "estimated_z": float(self.estimated_position[2]),
            "estimated_cell_id": self.estimated_cell_id,
            "estimated_px": float(self.estimated_moment[0]),
            "estimated_py": float(self.estimated_moment[1]),
            "estimated_pz": float(self.estimated_moment[2]),
            "residual_norm": self.residual_norm,
            "relative_residual": self.relative_residual,
            "localization_error": self.localization_error,
            "moment_relative_error": self.moment_relative_error,
            "moment_angle_error_deg": self.moment_angle_error_deg,
            "success": self.success,
        }


@dataclass(frozen=True)
class InverseBenchmarkResult:
    scenario: Any
    records: list[InverseBenchmarkRecord]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(isinstance(record, InverseBenchmarkRecord) for record in self.records):
            raise TypeError("records must contain InverseBenchmarkRecord objects")
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
        localization_errors = np.asarray([record.localization_error for record in self.records], dtype=float)
        angle_errors = np.asarray([record.moment_angle_error_deg for record in self.records], dtype=float)
        relative_residuals = np.asarray([record.relative_residual for record in self.records], dtype=float)
        successes = [record.success for record in self.records if record.success is not None]
        summary = {
            "scenario_name": scenario_name,
            "num_records": len(self.records),
            "measurement_kinds": list(dict.fromkeys(record.measurement_kind for record in self.records)),
            "electrode_set_names": list(dict.fromkeys(record.electrode_set_name for record in self.records)),
            "noise_model_names": list(dict.fromkeys(record.noise_model_name for record in self.records)),
            "lambda_reg": getattr(self.scenario, "lambda_reg", None),
            "localization_error_mean": float(np.nanmean(localization_errors)) if localization_errors.size else np.nan,
            "localization_error_median": float(np.nanmedian(localization_errors)) if localization_errors.size else np.nan,
            "localization_error_p90": float(np.nanpercentile(localization_errors, 90)) if localization_errors.size else np.nan,
            "moment_angle_error_mean": float(np.nanmean(angle_errors)) if angle_errors.size else np.nan,
            "relative_residual_mean": float(np.nanmean(relative_residuals)) if relative_residuals.size else np.nan,
            "metadata": dict(self.metadata),
        }
        if successes:
            summary["success_rate"] = float(np.mean(successes))
        return summary
