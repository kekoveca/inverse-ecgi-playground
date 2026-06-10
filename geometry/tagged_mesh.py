from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .mesh_model import MeshData


@dataclass(frozen=True)
class TaggedMesh:
    """Mesh loaded from Gmsh/meshio with physical tags.

    This container is useful when the mesh contains several cell blocks
    such as triangles for the computational domain and lines for boundaries
    in 2D, or tetrahedra for the volume and triangles for boundaries in 3D.

    Attributes
    ----------
    dim:
        Geometric dimension used by the benchmark (2 or 3).
    coords:
        Node coordinates with shape ``(n_nodes, dim)``.
    cells:
        Mapping from meshio cell type to connectivity array. Examples:
        ``{"triangle": (n_tri, 3), "line": (n_line, 2)}`` in 2D or
        ``{"tetra": (n_tet, 4), "triangle": (n_tri, 3)}`` in 3D.
    cell_tags:
        Mapping from cell type to physical tag array. For every key in
        ``cells``, ``cell_tags[key]`` has length ``len(cells[key])``.
    field_data:
        Gmsh physical group metadata: ``name -> (dim, tag)``.
    metadata:
        Optional free-form metadata.
    """

    dim: int
    coords: np.ndarray
    cells: dict[str, np.ndarray]
    cell_tags: dict[str, np.ndarray]
    field_data: dict[str, tuple[int, int]]  # name -> (dim, tag)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=float)
        if coords.ndim != 2:
            raise ValueError("coords must have shape (n_nodes, dim)")
        if self.dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if coords.shape[1] != self.dim:
            raise ValueError(f"coords second dimension must equal dim={self.dim}")

        cells: dict[str, np.ndarray] = {}
        for cell_type, conn in self.cells.items():
            arr = np.asarray(conn, dtype=np.int64)
            if arr.ndim != 2:
                raise ValueError(f"cells[{cell_type!r}] must be a 2D array")
            if arr.size > 0:
                if arr.min() < 0 or arr.max() >= coords.shape[0]:
                    raise ValueError(f"cells[{cell_type!r}] contain invalid node indices")
            cells[cell_type] = arr

        cell_tags: dict[str, np.ndarray] = {}
        for cell_type, conn in cells.items():
            if cell_type in self.cell_tags:
                tags = np.asarray(self.cell_tags[cell_type], dtype=np.int64)
            else:
                tags = np.zeros(conn.shape[0], dtype=np.int64)
            if tags.ndim != 1 or tags.shape[0] != conn.shape[0]:
                raise ValueError(f"cell_tags[{cell_type!r}] must have shape ({conn.shape[0]},), " f"got {tags.shape}")
            cell_tags[cell_type] = tags

        field_data: dict[str, tuple[int, int]] = {}
        for name, value in self.field_data.items():
            physical_dim, tag = value
            field_data[str(name)] = (int(physical_dim), int(tag))

        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "cell_tags", cell_tags)
        object.__setattr__(self, "field_data", field_data)

    @property
    def num_points(self) -> int:
        return int(self.coords.shape[0])

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
        if cell_type not in self.cells:
            raise KeyError(f"Cell type {cell_type!r} not found. Available: {list(self.cells.keys())}")
        block = self.cells[cell_type]
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

    def to_mesh_data(self, cell_type: str, name: str | None = None, physical_name: str | None = None) -> MeshData:
        """Convert one cell block to the lightweight ``MeshData`` container.

        Parameters
        ----------
        cell_type:
            meshio cell type to convert, for example ``"triangle"`` or ``"tetra"``.
        name:
            Optional name for the resulting ``MeshData``.
        physical_name:
            Optional Gmsh physical group name. If provided, only cells with this
            physical tag are included.
        """
        cells = self.cell_block(cell_type, physical_name=physical_name)
        metadata = dict(self.metadata)
        metadata.update(
            {
                "source": "TaggedMesh",
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
        return MeshData(points=self.coords, cells=cells, cell_type=cell_type, name=mesh_name, metadata=metadata)


def _field_data_to_tuples(field_data: dict[str, Any]) -> dict[str, tuple[int, int]]:
    converted: dict[str, tuple[int, int]] = {}
    for name, value in field_data.items():
        arr = np.asarray(value, dtype=int).ravel()
        if arr.size < 2:
            raise ValueError(f"field_data[{name!r}] must contain at least tag and dimension")
        tag = int(arr[0])
        dim = int(arr[1])
        converted[str(name)] = (dim, tag)
    return converted


def read_gmsh_meshio(path: str | Path, dim: int = 2) -> TaggedMesh:
    """Read a Gmsh mesh through meshio and keep physical tags.

    The function supports the common cases needed for the benchmark:

    - 2D: ``triangle`` domain cells and ``line`` boundary cells;
    - 3D: ``tetra`` volume cells and ``triangle`` boundary cells.

    If the file does not contain Gmsh physical tags, zero tags are created for
    each loaded cell block.
    """
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("read_gmsh_meshio requires the optional dependency 'meshio'") from exc

    m = meshio.read(Path(path))
    coords = np.asarray(m.points[:, :dim], dtype=float)

    expected_nodes = {"line": 2, "triangle": 3, "tetra": 4}
    cells: dict[str, np.ndarray] = {}
    for cell_type in ("tetra", "triangle", "line"):
        if cell_type in m.cells_dict:
            arr = np.asarray(m.cells_dict[cell_type], dtype=np.int64)
            if arr.shape[1] != expected_nodes[cell_type]:
                raise ValueError(f"Unexpected node count for cell type {cell_type!r}: {arr.shape[1]}")
            cells[cell_type] = arr

    cell_tags: dict[str, np.ndarray] = {}
    phys_data = m.cell_data_dict.get("gmsh:physical", {}) if hasattr(m, "cell_data_dict") else {}
    for cell_type, conn in cells.items():
        if cell_type in phys_data:
            cell_tags[cell_type] = np.asarray(phys_data[cell_type], dtype=np.int64)
        else:
            cell_tags[cell_type] = np.zeros(conn.shape[0], dtype=np.int64)

    return TaggedMesh(
        dim=dim,
        coords=coords,
        cells=cells,
        cell_tags=cell_tags,
        field_data=_field_data_to_tuples(m.field_data),
        metadata={"path": str(path), "reader": "meshio"},
    )


# Backward-compatible alias
Mesh = TaggedMesh
