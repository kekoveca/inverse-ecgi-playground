from __future__ import annotations

import csv
import json
from pathlib import Path

from .io import _json_safe
from .inverse_results import InverseBenchmarkResult


def _scenario_config(result: InverseBenchmarkResult) -> dict:
    scenario = result.scenario
    if hasattr(scenario, "to_config_dict"):
        return scenario.to_config_dict()
    if isinstance(scenario, dict):
        return dict(scenario)
    raise TypeError("result.scenario must provide to_config_dict() or be a dict")


def save_inverse_benchmark_result(result: InverseBenchmarkResult, output_dir, save_maps: bool = False) -> Path:
    """Save inverse benchmark config, scalar records and summary."""
    if save_maps:
        raise NotImplementedError("saving inverse residual/moment maps is not implemented")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "inverse_config.json").write_text(
        json.dumps(_json_safe(_scenario_config(result)), indent=2),
        encoding="utf-8",
    )
    (output_path / "inverse_summary.json").write_text(
        json.dumps(_json_safe(result.summary()), indent=2),
        encoding="utf-8",
    )

    rows = result.to_rows()
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_path / "inverse_records.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))
    return output_path
