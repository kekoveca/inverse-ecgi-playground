from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from green import GreenTransferMatrix

from .results import ForwardBenchmarkResult


@dataclass(frozen=True)
class InverseBenchmarkScenario:
    """Configuration for inverse benchmark over forward benchmark records."""

    name: str
    forward_result: ForwardBenchmarkResult | dict
    transfer_matrix: GreenTransferMatrix
    lambda_reg: float = 0.0
    localization_threshold: float | None = None
    use_noisy_measurements: bool = True
    use_clean_measurements: bool = True
    reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lambda_reg", float(self.lambda_reg))
        if self.localization_threshold is not None:
            object.__setattr__(self, "localization_threshold", float(self.localization_threshold))
        object.__setattr__(self, "use_noisy_measurements", bool(self.use_noisy_measurements))
        object.__setattr__(self, "use_clean_measurements", bool(self.use_clean_measurements))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def records(self):
        if isinstance(self.forward_result, ForwardBenchmarkResult):
            return self.forward_result.records
        if isinstance(self.forward_result, dict) and "records" in self.forward_result:
            return list(self.forward_result["records"])
        raise TypeError("forward_result must be ForwardBenchmarkResult or a dict with records")

    def validate(self) -> None:
        if not str(self.name).strip():
            raise ValueError("inverse benchmark scenario name must not be empty")
        if not isinstance(self.transfer_matrix, GreenTransferMatrix):
            raise TypeError("transfer_matrix must be a GreenTransferMatrix")
        if self.transfer_matrix.num_candidates < 1:
            raise ValueError("transfer_matrix must contain at least one candidate")
        if self.transfer_matrix.num_measurements < 1:
            raise ValueError("transfer_matrix must contain at least one measurement channel")
        if self.lambda_reg < 0.0:
            raise ValueError("lambda_reg must be non-negative")
        if not self.use_clean_measurements and not self.use_noisy_measurements:
            raise ValueError("at least one of use_clean_measurements/use_noisy_measurements must be True")
        records = self.records
        if not records:
            raise ValueError("forward_result.records must not be empty")

        expected = self.transfer_matrix.num_measurements
        lengths = set()
        electrode_sets = set()
        for record in records:
            clean = np.asarray(record.clean_measurements, dtype=float)
            noisy = np.asarray(record.noisy_measurements, dtype=float)
            lengths.add(clean.size)
            lengths.add(noisy.size)
            electrode_sets.add(record.electrode_set_name)
            if clean.shape != (expected,) or noisy.shape != (expected,):
                raise ValueError(
                    "forward record measurement length does not match GreenTransferMatrix; "
                    "filter records to one electrode_set/transfer matrix before running inverse benchmark"
                )
        if lengths != {expected}:
            raise ValueError("forward records contain measurement vectors with inconsistent lengths")
        if len(electrode_sets) > 1:
            raise ValueError(
                "InverseBenchmarkScenario currently supports one electrode_set per GreenTransferMatrix; "
                "use filter_forward_result_by_electrode_set first"
            )

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lambda_reg": self.lambda_reg,
            "localization_threshold": self.localization_threshold,
            "use_noisy_measurements": self.use_noisy_measurements,
            "use_clean_measurements": self.use_clean_measurements,
            "num_forward_records": len(self.records),
            "transfer_num_candidates": self.transfer_matrix.num_candidates,
            "transfer_num_measurements": self.transfer_matrix.num_measurements,
            "reference": self.reference,
            "metadata": dict(self.metadata),
        }


def filter_forward_result_by_electrode_set(
    forward_result: ForwardBenchmarkResult,
    electrode_set_name: str,
) -> ForwardBenchmarkResult:
    """Return a new ForwardBenchmarkResult containing only one electrode subset."""
    if not isinstance(forward_result, ForwardBenchmarkResult):
        raise TypeError("forward_result must be a ForwardBenchmarkResult")
    records = [record for record in forward_result.records if record.electrode_set_name == electrode_set_name]
    if not records:
        raise ValueError(f"no forward benchmark records found for electrode_set_name={electrode_set_name!r}")
    scenario = forward_result.scenario
    if hasattr(scenario, "to_config_dict"):
        scenario = scenario.to_config_dict()
    if isinstance(scenario, dict):
        scenario = dict(scenario)
        scenario["filtered_electrode_set_name"] = electrode_set_name
        scenario["num_records"] = len(records)
    return ForwardBenchmarkResult(
        scenario=scenario,
        records=records,
        metadata={**forward_result.metadata, "filtered_electrode_set_name": electrode_set_name},
    )
