from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geometry import ElectrodeSet


@dataclass(frozen=True)
class ElectrodeSubset:
    name: str
    electrodes: ElectrodeSet
    indices: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("ElectrodeSubset name must not be empty")
        if not isinstance(self.electrodes, ElectrodeSet):
            raise TypeError("electrodes must be an ElectrodeSet")
        indices = None if self.indices is None else np.asarray(self.indices, dtype=np.int64)
        if indices is not None:
            if indices.shape != (self.electrodes.num_electrodes,):
                raise ValueError("indices length must match subset electrode count")
            if np.unique(indices).size != indices.size:
                raise ValueError("indices must be unique")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def __len__(self) -> int:
        return self.electrodes.num_electrodes

    @property
    def labels(self) -> list[str]:
        return list(self.electrodes.labels)

    @property
    def positions(self) -> np.ndarray:
        return self.electrodes.positions

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_electrodes": len(self),
            "labels": self.labels,
            "indices": None if self.indices is None else self.indices.tolist(),
            "metadata": dict(self.metadata),
        }


def _validate_selection(electrodes: ElectrodeSet, indices) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("indices must be a non-empty one-dimensional sequence")
    if np.unique(selected).size != selected.size:
        raise ValueError("indices must be unique")
    if selected.min() < 0 or selected.max() >= electrodes.num_electrodes:
        raise ValueError("indices contain values outside the electrode set")
    return selected


def make_all_electrodes_subset(electrodes: ElectrodeSet, name: str = "all") -> ElectrodeSubset:
    indices = np.arange(electrodes.num_electrodes, dtype=np.int64)
    return select_electrodes_by_indices(electrodes, indices, name=name)


def select_electrodes_by_indices(electrodes: ElectrodeSet, indices, name: str | None = None) -> ElectrodeSubset:
    """Select electrodes while preserving the provided index order."""
    selected = _validate_selection(electrodes, indices)
    subset_name = name or f"indices_{selected.size}"
    subset = ElectrodeSet(
        positions=electrodes.positions[selected],
        labels=[electrodes.labels[index] for index in selected],
        name=subset_name,
        metadata={"parent_electrode_set": electrodes.name},
    )
    return ElectrodeSubset(name=subset_name, electrodes=subset, indices=selected)


def _validate_n(electrodes: ElectrodeSet, n: int) -> int:
    n = int(n)
    if n < 1:
        raise ValueError("n must be positive")
    if n > electrodes.num_electrodes:
        raise ValueError("n exceeds number of electrodes")
    return n


def select_random_electrodes(
    electrodes: ElectrodeSet,
    n: int,
    seed: int = 0,
    name: str | None = None,
) -> ElectrodeSubset:
    n = _validate_n(electrodes, n)
    indices = np.random.default_rng(seed).choice(electrodes.num_electrodes, size=n, replace=False)
    subset = select_electrodes_by_indices(electrodes, indices, name=name or f"random_{n}")
    metadata = {"selection": "random_without_replacement", "seed": int(seed)}
    return ElectrodeSubset(subset.name, subset.electrodes, subset.indices, metadata)


def select_farthest_point_electrodes(
    electrodes: ElectrodeSet,
    n: int,
    seed: int = 0,
    start_index: int | None = None,
    name: str | None = None,
) -> ElectrodeSubset:
    """Select a spatially spread subset by greedy farthest-point sampling."""
    n = _validate_n(electrodes, n)
    rng = np.random.default_rng(seed)
    if start_index is None:
        start = int(rng.integers(electrodes.num_electrodes))
    else:
        start = int(start_index)
        if start < 0 or start >= electrodes.num_electrodes:
            raise ValueError("start_index is outside the electrode set")

    positions = electrodes.positions
    selected = [start]
    min_distances = np.linalg.norm(positions - positions[start], axis=1)
    min_distances[start] = -np.inf
    while len(selected) < n:
        next_index = int(np.argmax(min_distances))
        selected.append(next_index)
        distances = np.linalg.norm(positions - positions[next_index], axis=1)
        min_distances = np.minimum(min_distances, distances)
        min_distances[np.asarray(selected, dtype=np.int64)] = -np.inf

    subset = select_electrodes_by_indices(electrodes, selected, name=name or f"farthest_{n}")
    metadata = {"selection": "farthest_point", "seed": int(seed), "start_index": start}
    return ElectrodeSubset(subset.name, subset.electrodes, subset.indices, metadata)
