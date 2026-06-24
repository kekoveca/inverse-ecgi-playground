from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .transfer_matrix import GreenTransferMatrix


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def save_green_transfer_matrix(transfer_matrix: GreenTransferMatrix, path) -> Path:
    """Save a transfer tensor and compact metadata to an unpickled NPZ cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(transfer_matrix.metadata, default=_json_default, sort_keys=True)
    np.savez_compressed(
        path,
        A=transfer_matrix.A,
        candidate_points=transfer_matrix.candidate_points,
        candidate_cell_ids=transfer_matrix.candidate_cell_ids,
        measurement_row_indices=transfer_matrix.measurement_row_indices,
        sign=np.asarray(transfer_matrix.sign, dtype=float),
        metadata_json=np.asarray(metadata_json),
    )
    return path


def load_green_transfer_matrix(path) -> GreenTransferMatrix:
    """Load a transfer tensor saved by :func:`save_green_transfer_matrix`."""
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return GreenTransferMatrix(
            A=data["A"],
            candidate_points=data["candidate_points"],
            candidate_cell_ids=data["candidate_cell_ids"],
            sign=float(data["sign"].item()),
            metadata=metadata,
            measurement_row_indices=data["measurement_row_indices"] if "measurement_row_indices" in data.files else None,
        )
