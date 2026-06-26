from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .timer import PerformanceTimer


def _json_ready(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def save_timing_json(timer: PerformanceTimer, path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": timer.summary(),
        "records": timer.to_rows(),
    }
    output_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def save_timing_csv(timer: PerformanceTimer, path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = timer.to_rows()
    fieldnames = _fieldnames(rows)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return output_path


def _fieldnames(rows: Iterable[dict]) -> list[str]:
    fields = ["name", "elapsed_s"]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def format_timing_table(timer: PerformanceTimer) -> str:
    """Return a markdown timing table."""
    lines = ["| Stage | Time, s | Metadata |", "| --- | ---: | --- |"]
    for record in timer.records:
        metadata = ", ".join(f"{key}={value}" for key, value in record.metadata.items())
        lines.append(f"| `{record.name}` | {record.elapsed_s:.6g} | {metadata} |")
    return "\n".join(lines)
