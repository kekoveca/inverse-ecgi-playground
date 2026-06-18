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
    validates mesh-like data that later can be used externally.

    ``MeshData`` can represent either one active cell block or a Gmsh/meshio
    mesh with several cell blocks. In the multi-block case ``cells`` and
    ``cell_type`` identify the active block, while ``cell_blocks`` stores all
    available blocks and ``cell_tags`` stores physical tags per block.

    Node and cell ids belong to this container's ordering and must not be
    treated as DOLFINx dof/cell ids. ``field_data`` uses ``name -> (dim, tag)``;
    ``read_gmsh_meshio`` converts meshio's ``(tag, dim)`` values on input.
    """

    points: np.ndarray
    cells: np.ndarray
    cell_type: CellType = "tetra"
    name: str = "mesh"
    metadata: dict[str, Any] = field(default_factory=dict)
    cell_tags: dict[str, np.ndarray] | np.ndarray | None = None
    field_data: dict[str, tuple[int, int]] = field(default_factory=dict)  # name -> (dim, tag)
    cell_blocks: dict[str, np.ndarray] | None = None

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

        cell_blocks = self._normalize_cell_blocks(points=points, active_cells=cells)
        cell_tags = self._normalize_cell_tags(cell_blocks=cell_blocks)
        field_data = self._normalize_field_data(self.field_data)

        object.__setattr__(self, "points", points)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "cell_blocks", cell_blocks)
        object.__setattr__(self, "cell_tags", cell_tags)
        object.__setattr__(self, "field_data", field_data)

    def _normalize_cell_blocks(self, *, points: np.ndarray, active_cells: np.ndarray) -> dict[str, np.ndarray]:
        raw_blocks = self.cell_blocks or {self.cell_type: active_cells}
        cell_blocks: dict[str, np.ndarray] = {}
        expected_nodes = {"line": 2, "triangle": 3, "tetra": 4}

        for cell_type, conn in raw_blocks.items():
            if cell_type not in expected_nodes:
                raise ValueError(f"Unsupported cell_type={cell_type!r}; expected one of {sorted(expected_nodes)}")
            arr = np.asarray(conn, dtype=np.int64)
            if arr.ndim != 2:
                raise ValueError(f"cell_blocks[{cell_type!r}] must be a 2D array")
            if arr.shape[1] != expected_nodes[cell_type]:
                raise ValueError(
                    f"cell_type={cell_type!r} requires {expected_nodes[cell_type]} nodes per cell, "
                    f"got {arr.shape[1]}"
                )
            if arr.size > 0:
                if arr.min() < 0 or arr.max() >= points.shape[0]:
                    raise ValueError(f"cell_blocks[{cell_type!r}] contain node indices outside points array")
            cell_blocks[cell_type] = arr

        if self.cell_type not in cell_blocks:
            cell_blocks[self.cell_type] = active_cells
        return cell_blocks

    def _normalize_cell_tags(self, *, cell_blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.cell_tags is None:
            raw_tags: dict[str, np.ndarray] = {}
        elif isinstance(self.cell_tags, dict):
            raw_tags = self.cell_tags
        else:
            raw_tags = {self.cell_type: np.asarray(self.cell_tags, dtype=np.int64)}

        cell_tags: dict[str, np.ndarray] = {}
        for cell_type, conn in cell_blocks.items():
            if cell_type in raw_tags:
                tags = np.asarray(raw_tags[cell_type], dtype=np.int64)
            else:
                tags = np.zeros(conn.shape[0], dtype=np.int64)
            if tags.ndim != 1 or tags.shape[0] != conn.shape[0]:
                raise ValueError(f"cell_tags[{cell_type!r}] must have shape ({conn.shape[0]},), got {tags.shape}")
            cell_tags[cell_type] = tags
        return cell_tags

    @staticmethod
    def _normalize_field_data(field_data: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
        converted: dict[str, tuple[int, int]] = {}
        for name, value in field_data.items():
            arr = np.asarray(value, dtype=int).ravel()
            if arr.size < 2:
                raise ValueError(f"field_data[{name!r}] must contain at least dimension and tag")
            converted[str(name)] = (int(arr[0]), int(arr[1]))
        return converted

    @classmethod
    def from_cell_blocks(
        cls,
        points: np.ndarray,
        cell_blocks: dict[str, np.ndarray],
        *,
        cell_tags: dict[str, np.ndarray] | None = None,
        field_data: dict[str, tuple[int, int]] | None = None,
        primary_cell_type: CellType | None = None,
        name: str = "mesh",
        metadata: dict[str, Any] | None = None,
    ) -> "MeshData":
        if not cell_blocks:
            raise ValueError("cell_blocks must contain at least one cell block")
        if primary_cell_type is None:
            for candidate in ("tetra", "triangle", "line"):
                if candidate in cell_blocks:
                    primary_cell_type = candidate  # type: ignore[assignment]
                    break
        if primary_cell_type is None or primary_cell_type not in cell_blocks:
            raise ValueError("primary_cell_type must name an existing cell block")
        return cls(
            points=points,
            cells=cell_blocks[primary_cell_type],
            cell_type=primary_cell_type,
            name=name,
            metadata=metadata or {},
            cell_tags=cell_tags,
            field_data=field_data or {},
            cell_blocks=cell_blocks,
        )

    @property
    def coords(self) -> np.ndarray:
        """Coordinate alias for meshio-style workflows."""
        return self.points

    @property
    def dim(self) -> int:
        """Backward-compatible geometric dimension alias."""
        return self.geometric_dim

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
            cell_tags=self.cell_tags,
            field_data=self.field_data,
            cell_blocks=self.cell_blocks,
        )

    def physical_tag(self, name: str) -> int:
        """Return integer physical tag by Gmsh physical group name."""
        if name not in self.field_data:
            available = list(self.field_data.keys())
            raise KeyError(f"Physical group {name!r} not found. Available: {available}")
        return int(self.field_data[name][1])

    def physical_dimension(self, name: str) -> int:
        """Return Gmsh physical dimension by physical group name."""
        if name not in self.field_data:
            available = list(self.field_data.keys())
            raise KeyError(f"Physical group {name!r} not found. Available: {available}")
        return int(self.field_data[name][0])

    def cell_block(self, cell_type: str, physical_name: str | None = None) -> np.ndarray:
        """Return cells of a given type, optionally filtered by physical group."""
        if cell_type not in self.cell_blocks:
            raise KeyError(f"Cell type {cell_type!r} not found. Available: {list(self.cell_blocks.keys())}")
        block = self.cell_blocks[cell_type]
        if physical_name is None:
            return block
        tag = self.physical_tag(physical_name)
        tags = self.cell_tags[cell_type]
        return block[tags == tag]

    def tags_for(self, cell_type: str) -> np.ndarray:
        """Return physical tags for a cell block."""
        if cell_type not in self.cell_tags:
            raise KeyError(f"Cell type {cell_type!r} not found. Available: {list(self.cell_tags.keys())}")
        return self.cell_tags[cell_type]

    def to_mesh_data(self, cell_type: str, name: str | None = None, physical_name: str | None = None) -> "MeshData":
        """Return one cell block as an active single-block ``MeshData`` view."""
        cells = self.cell_block(cell_type, physical_name=physical_name)
        tags = self.tags_for(cell_type)
        if physical_name is not None:
            tag = self.physical_tag(physical_name)
            tags = tags[tags == tag]

        metadata = dict(self.metadata)
        metadata.update(
            {
                "source": "MeshData",
                "cell_type": cell_type,
                "physical_name": physical_name,
                "field_data": self.field_data,
            }
        )
        if physical_name is not None:
            metadata.update(
                {
                    "physical_dimension": self.physical_dimension(physical_name),
                    "physical_tag": self.physical_tag(physical_name),
                }
            )
        mesh_name = name or (physical_name if physical_name is not None else cell_type)
        return MeshData(
            points=self.points,
            cells=cells,
            cell_type=cell_type,
            name=mesh_name,
            metadata=metadata,
            cell_tags={cell_type: tags},
            field_data=self.field_data,
            cell_blocks={cell_type: cells},
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

    Optional fields: cell_type, name, metadata, cell_tags, field_data,
    cell_blocks.
    """
    data = np.load(Path(path), allow_pickle=True)
    cell_type = str(data["cell_type"]) if "cell_type" in data else "tetra"
    name = str(data["name"]) if "name" in data else Path(path).stem
    metadata = data["metadata"].item() if "metadata" in data else {}
    cell_tags = data["cell_tags"].item() if "cell_tags" in data else None
    field_data = data["field_data"].item() if "field_data" in data else {}
    cell_blocks = data["cell_blocks"].item() if "cell_blocks" in data else None
    return MeshData(
        points=data["points"],
        cells=data["cells"],
        cell_type=cell_type,
        name=name,
        metadata=metadata,
        cell_tags=cell_tags,
        field_data=field_data,
        cell_blocks=cell_blocks,
    )


def save_npz_mesh(mesh: MeshData, path: str | Path) -> None:
    np.savez_compressed(
        Path(path),
        points=mesh.points,
        cells=mesh.cells,
        cell_type=np.array(mesh.cell_type),
        name=np.array(mesh.name),
        metadata=np.array(mesh.metadata, dtype=object),
        cell_tags=np.array(mesh.cell_tags, dtype=object),
        field_data=np.array(mesh.field_data, dtype=object),
        cell_blocks=np.array(mesh.cell_blocks, dtype=object),
    )


def _field_data_to_tuples(field_data: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Convert meshio ``name -> (tag, dim)`` to internal ``name -> (dim, tag)``."""
    converted: dict[str, tuple[int, int]] = {}
    for name, value in field_data.items():
        arr = np.asarray(value, dtype=int).ravel()
        if arr.size < 2:
            raise ValueError(f"field_data[{name!r}] must contain at least tag and dimension")
        tag = int(arr[0])
        dim = int(arr[1])
        converted[str(name)] = (dim, tag)
    return converted


def read_gmsh_meshio(path: str | Path, dim: int = 2) -> MeshData:
    """Read a Gmsh mesh through meshio and keep physical tags.

    meshio reports ``field_data`` as ``name -> (tag, dim)``. ``MeshData`` stores
    it internally in the Gmsh API order ``name -> (dim, tag)``.
    """
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("read_gmsh_meshio requires the optional dependency 'meshio'") from exc

    m = meshio.read(Path(path))
    coords = np.asarray(m.points[:, :dim], dtype=float)

    expected_nodes = {"line": 2, "triangle": 3, "tetra": 4}
    cell_blocks: dict[str, np.ndarray] = {}
    for cell_type in ("tetra", "triangle", "line"):
        if cell_type in m.cells_dict:
            arr = np.asarray(m.cells_dict[cell_type], dtype=np.int64)
            if arr.shape[1] != expected_nodes[cell_type]:
                raise ValueError(f"Unexpected node count for cell type {cell_type!r}: {arr.shape[1]}")
            cell_blocks[cell_type] = arr

    cell_tags: dict[str, np.ndarray] = {}
    phys_data = m.cell_data_dict.get("gmsh:physical", {}) if hasattr(m, "cell_data_dict") else {}
    for cell_type, conn in cell_blocks.items():
        if cell_type in phys_data:
            cell_tags[cell_type] = np.asarray(phys_data[cell_type], dtype=np.int64)
        else:
            cell_tags[cell_type] = np.zeros(conn.shape[0], dtype=np.int64)

    return MeshData.from_cell_blocks(
        points=coords,
        cell_blocks=cell_blocks,
        cell_tags=cell_tags,
        field_data=_field_data_to_tuples(m.field_data),
        metadata={"path": str(path), "reader": "meshio"},
    )
