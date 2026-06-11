from __future__ import annotations

"""Backward-compatible tagged mesh imports.

Tagged mesh functionality now lives in ``geometry.mesh_model.MeshData``. This
module remains so older imports from ``geometry.tagged_mesh`` keep working.
"""

import numpy as np

from .mesh_model import MeshData, _field_data_to_tuples, read_gmsh_meshio


class TaggedMesh(MeshData):
    """Compatibility wrapper for the old ``TaggedMesh`` constructor.

    New code should use ``MeshData`` directly, especially
    ``MeshData.from_cell_blocks`` for multi-block Gmsh meshes.
    """

    def __init__(
        self,
        dim: int,
        coords: np.ndarray,
        cells: dict[str, np.ndarray],
        cell_tags: dict[str, np.ndarray] | None = None,
        field_data: dict[str, tuple[int, int]] | None = None,
        metadata: dict | None = None,
    ) -> None:
        points = np.asarray(coords, dtype=float)
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if points.ndim != 2:
            raise ValueError("coords must have shape (n_nodes, dim)")
        if points.shape[1] != dim:
            raise ValueError(f"coords second dimension must equal dim={dim}")
        mesh = MeshData.from_cell_blocks(
            points=points,
            cell_blocks=cells,
            cell_tags=cell_tags,
            field_data=field_data or {},
            metadata=metadata or {},
        )
        object.__setattr__(self, "points", mesh.points)
        object.__setattr__(self, "cells", mesh.cells)
        object.__setattr__(self, "cell_type", mesh.cell_type)
        object.__setattr__(self, "name", mesh.name)
        object.__setattr__(self, "metadata", mesh.metadata)
        object.__setattr__(self, "cell_tags", mesh.cell_tags)
        object.__setattr__(self, "field_data", mesh.field_data)
        object.__setattr__(self, "cell_blocks", mesh.cell_blocks)


Mesh = MeshData
