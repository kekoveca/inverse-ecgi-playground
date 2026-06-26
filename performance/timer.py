from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass(frozen=True)
class TimingRecord:
    """One named timing measurement."""

    name: str
    elapsed_s: float
    metadata: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        row = {"name": self.name, "elapsed_s": float(self.elapsed_s)}
        row.update(self.metadata)
        return row


class PerformanceTimer:
    """Collect named timing records with context-manager syntax."""

    def __init__(self) -> None:
        self.records: list[TimingRecord] = []

    @contextmanager
    def time(self, name: str, **metadata) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.add_record(name, perf_counter() - start, **metadata)

    def add_record(self, name: str, elapsed_s: float, **metadata) -> TimingRecord:
        record = TimingRecord(str(name), float(elapsed_s), dict(metadata))
        self.records.append(record)
        return record

    def to_rows(self) -> list[dict]:
        return [record.to_row() for record in self.records]

    def summary(self) -> dict:
        total = float(sum(record.elapsed_s for record in self.records))
        by_name: dict[str, float] = {}
        for record in self.records:
            by_name[record.name] = by_name.get(record.name, 0.0) + float(record.elapsed_s)
        return {
            "num_records": len(self.records),
            "total_recorded_s": total,
            "by_name": by_name,
        }
