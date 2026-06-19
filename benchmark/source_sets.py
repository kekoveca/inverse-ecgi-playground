from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from sources import PointDipole


def axis_moments(scale: float = 1.0) -> list[np.ndarray]:
    """Return x/y/z Cartesian dipole moments with the requested scale."""
    scale = float(scale)
    if not np.isfinite(scale):
        raise ValueError("scale must be finite")
    return [
        np.array([scale, 0.0, 0.0]),
        np.array([0.0, scale, 0.0]),
        np.array([0.0, 0.0, scale]),
    ]


def _normalize_moments(moments) -> list[np.ndarray]:
    raw = axis_moments() if moments is None else list(moments)
    if not raw:
        raise ValueError("moments must contain at least one vector")
    normalized = []
    for moment in raw:
        array = np.asarray(moment, dtype=float)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError("each moment must be a finite vector with shape (3,)")
        normalized.append(array.copy())
    return normalized


def _region_arrays(source_region) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.asarray(source_region.candidate_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("source_region.candidate_points must have shape (n_candidates, 3)")
    if points.shape[0] == 0:
        raise ValueError("source_region contains no candidate points")
    raw_ids = getattr(source_region, "candidate_cell_ids", None)
    if raw_ids is None:
        return points, None
    cell_ids = np.asarray(raw_ids, dtype=np.int64)
    if cell_ids.shape != (points.shape[0],):
        raise ValueError("candidate_cell_ids must match candidate_points")
    return points, cell_ids


@dataclass(frozen=True)
class SourceSet:
    name: str
    sources: list[PointDipole]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("SourceSet name must not be empty")
        if not all(isinstance(source, PointDipole) for source in self.sources):
            raise TypeError("sources must contain PointDipole objects")
        object.__setattr__(self, "sources", list(self.sources))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def __len__(self) -> int:
        return len(self.sources)

    def __iter__(self) -> Iterator[PointDipole]:
        return iter(self.sources)

    def to_table(self) -> list[dict[str, Any]]:
        rows = []
        for index, source in enumerate(self.sources):
            rows.append(
                {
                    "source_index": index,
                    "source_name": source.name,
                    "x": float(source.position[0]),
                    "y": float(source.position[1]),
                    "z": float(source.position[2]),
                    "px": float(source.moment[0]),
                    "py": float(source.moment[1]),
                    "pz": float(source.moment[2]),
                    "cell_id": source.cell_id,
                }
            )
        return rows


def _sources_from_indices(source_region, indices, moments, name: str, metadata: dict[str, Any]) -> SourceSet:
    points, cell_ids = _region_arrays(source_region)
    moments_list = _normalize_moments(moments)
    sources = []
    for candidate_index in np.asarray(indices, dtype=np.int64):
        for moment in moments_list:
            source = PointDipole(position=points[candidate_index], moment=moment)
            if cell_ids is not None:
                source = source.with_cell_id(int(cell_ids[candidate_index]))
            sources.append(source)
    return SourceSet(name=name, sources=sources, metadata=metadata)


def generate_sources_from_region(
    source_region,
    moments=None,
    max_positions: int | None = None,
    stride: int = 1,
    name: str = "source_region_sources",
) -> SourceSet:
    """Generate deterministic position/moment combinations from a source region."""
    points, _ = _region_arrays(source_region)
    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be positive")
    available = np.arange(0, points.shape[0], stride, dtype=np.int64)
    if max_positions is not None:
        max_positions = int(max_positions)
        if max_positions < 1:
            raise ValueError("max_positions must be positive")
        if max_positions < available.size:
            selection = np.linspace(0, available.size - 1, num=max_positions, dtype=np.int64)
            available = available[selection]
    return _sources_from_indices(
        source_region,
        available,
        moments,
        name,
        {"selection": "deterministic", "stride": stride, "max_positions": max_positions},
    )


def generate_random_sources_from_region(
    source_region,
    n_positions: int,
    moments=None,
    seed: int = 0,
    name: str = "random_sources",
) -> SourceSet:
    """Sample source positions without replacement using a reproducible seed."""
    points, _ = _region_arrays(source_region)
    n_positions = int(n_positions)
    if n_positions < 1:
        raise ValueError("n_positions must be positive")
    if n_positions > points.shape[0]:
        raise ValueError("n_positions exceeds number of source candidates")
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=n_positions, replace=False)
    return _sources_from_indices(
        source_region,
        indices,
        moments,
        name,
        {"selection": "random_without_replacement", "n_positions": n_positions, "seed": int(seed)},
    )
