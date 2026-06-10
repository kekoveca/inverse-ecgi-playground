from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

CellType = Literal["tetra", "triangle", "line"]


@dataclass(frozen=True)
class MeshData:
    """Lightweight mesh container independent of any FEM solver.

    The geometry layer does not assemble FEM matrices. It only stores and
    validates mesh-like data that later can be used externally
    """

    points: np.ndarray
    cells: np.ndarray
    cell_type: CellType = "tetra"
    name: str = "mesh"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        cells = np.asarray(self.cells, dtype=np.int64)

        if points.ndim != 2:
            raise ValueError("points must have shape (n_points, geometric_dim)")
        if points.shape[1] not in (2, 3):
            raise ValueError("points must be 2D or 3D coordinates")
        if cells.ndim != 2:
            raise ValueError("cells must have shape (n_cells, nodes_per_cell)")
        if cells.size > 0:
            if cells.min() < 0 or cells.max() >= points.shape[0]:
                raise ValueError("cells contain node indices outside points array")

        expected_nodes = {"line": 2, "triangle": 3, "tetra": 4}[self.cell_type]
        if cells.shape[1] != expected_nodes:
            raise ValueError(
                f"cell_type={self.cell_type!r} requires {expected_nodes} nodes per cell, " f"got {cells.shape[1]}"
            )

        object.__setattr__(self, "points", points)
        object.__setattr__(self, "cells", cells)

    @property
    def geometric_dim(self) -> int:
        return int(self.points.shape[1])

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def num_cells(self) -> int:
        return int(self.cells.shape[0])

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)

    def cell_centers(self, cell_ids: np.ndarray | None = None) -> np.ndarray:
        if cell_ids is None:
            selected = self.cells
        else:
            selected = self.cells[np.asarray(cell_ids, dtype=np.int64)]
        return self.points[selected].mean(axis=1)

    def with_metadata(self, **metadata: Any) -> "MeshData":
        new_metadata = dict(self.metadata)
        new_metadata.update(metadata)
        return MeshData(
            points=self.points,
            cells=self.cells,
            cell_type=self.cell_type,
            name=self.name,
            metadata=new_metadata,
        )


@dataclass(frozen=True)
class MeshQualityReport:
    num_points: int
    num_cells: int
    geometric_dim: int
    min_cell_volume: float | None
    max_cell_volume: float | None
    mean_cell_volume: float | None
    num_degenerate_cells: int


def tetra_volumes(mesh: MeshData) -> np.ndarray:
    """Return signed tetra volumes for a 3D tetra mesh."""
    if mesh.cell_type != "tetra":
        raise ValueError("tetra_volumes requires cell_type='tetra'")
    if mesh.geometric_dim != 3:
        raise ValueError("tetra_volumes requires 3D points")

    p = mesh.points[mesh.cells]
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    c = p[:, 3] - p[:, 0]
    return np.einsum("ij,ij->i", np.cross(a, b), c) / 6.0


def quality_report(mesh: MeshData, eps: float = 1e-14) -> MeshQualityReport:
    if mesh.cell_type == "tetra" and mesh.geometric_dim == 3 and mesh.num_cells > 0:
        vols = np.abs(tetra_volumes(mesh))
        min_vol = float(vols.min())
        max_vol = float(vols.max())
        mean_vol = float(vols.mean())
        num_degenerate = int(np.count_nonzero(vols <= eps))
    else:
        min_vol = max_vol = mean_vol = None
        num_degenerate = 0

    return MeshQualityReport(
        num_points=mesh.num_points,
        num_cells=mesh.num_cells,
        geometric_dim=mesh.geometric_dim,
        min_cell_volume=min_vol,
        max_cell_volume=max_vol,
        mean_cell_volume=mean_vol,
        num_degenerate_cells=num_degenerate,
    )


def load_npz_mesh(path: str | Path) -> MeshData:
    """Load a simple mesh stored as npz with arrays: points, cells.

    Optional fields: cell_type, name, metadata.
    """
    data = np.load(Path(path), allow_pickle=True)
    cell_type = str(data["cell_type"]) if "cell_type" in data else "tetra"
    name = str(data["name"]) if "name" in data else Path(path).stem
    metadata = data["metadata"].item() if "metadata" in data else {}
    return MeshData(points=data["points"], cells=data["cells"], cell_type=cell_type, name=name, metadata=metadata)


def save_npz_mesh(mesh: MeshData, path: str | Path) -> None:
    np.savez_compressed(
        Path(path),
        points=mesh.points,
        cells=mesh.cells,
        cell_type=np.array(mesh.cell_type),
        name=np.array(mesh.name),
        metadata=np.array(mesh.metadata, dtype=object),
    )
