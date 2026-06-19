from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .results import ForwardBenchmarkResult


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else "nan"
    return value


def _scenario_config(result: ForwardBenchmarkResult) -> dict[str, Any]:
    scenario = result.scenario
    if hasattr(scenario, "to_config_dict"):
        return scenario.to_config_dict()
    if isinstance(scenario, dict):
        return dict(scenario)
    raise TypeError("result.scenario must provide to_config_dict() or be a dict")


def save_forward_benchmark_result(result: ForwardBenchmarkResult, output_dir) -> Path:
    """Save benchmark config, scalar records, arrays and summary."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "config.json").write_text(
        json.dumps(_json_safe(_scenario_config(result)), indent=2),
        encoding="utf-8",
    )
    (output_path / "summary.json").write_text(
        json.dumps(_json_safe(result.summary()), indent=2),
        encoding="utf-8",
    )

    rows = result.to_rows()
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_path / "records.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))

    arrays = {}
    for index, record in enumerate(result.records):
        arrays[f"clean_{index:06d}"] = record.clean_measurements
        arrays[f"noisy_{index:06d}"] = record.noisy_measurements
        arrays[f"noise_{index:06d}"] = record.noise
    np.savez_compressed(output_path / "measurements.npz", **arrays)
    return output_path


def load_records_csv(path) -> list[dict[str, str]]:
    """Load scalar CSV rows without reconstructing benchmark records."""
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))
