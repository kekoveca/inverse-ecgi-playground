from __future__ import annotations

from typing import Any

import numpy as np

from geometry import MeshData

from ._imports import require_fenicsx


def infer_cell_type(mesh: MeshData) -> str:
    """Return the basix/UFL cell name for a ``MeshData`` simplex mesh."""
    if mesh.cell_type == "tetra":
        if mesh.geometric_dim != 3:
            raise ValueError("A tetra mesh must have 3D coordinates")
        return "tetrahedron"
    if mesh.cell_type == "triangle":
        if mesh.geometric_dim not in (2, 3):
            raise ValueError("A triangle mesh must have 2D or 3D coordinates")
        return "triangle"
    raise ValueError(f"Unsupported FEM cell_type={mesh.cell_type!r}; expected 'triangle' or 'tetra'")


def create_dolfinx_mesh(mesh: MeshData, comm: Any | None = None):
    """Convert ``geometry.MeshData`` to a DOLFINx mesh.

    This is intentionally kept outside the geometry layer so that geometry has
    no dependency on DOLFINx.
    """
    fx = require_fenicsx()
    MPI = fx["MPI"]
    basix_ufl = fx["basix_ufl"]
    dmesh = fx["dmesh"]
    ufl = fx["ufl"]

    if mesh.num_cells == 0:
        raise ValueError("Cannot create a DOLFINx mesh from MeshData with zero cells")

    if comm is None:
        comm = MPI.COMM_WORLD

    cell_name = infer_cell_type(mesh)
    gdim = mesh.geometric_dim

    # DOLFINx create_mesh expects a UFL mesh with a coordinate element.
    coordinate_element = basix_ufl.element("Lagrange", cell_name, 1, shape=(gdim,))
    domain = ufl.Mesh(coordinate_element)

    cells = np.asarray(mesh.cells, dtype=np.int64)
    points = np.asarray(mesh.points, dtype=np.float64)
    return dmesh.create_mesh(comm, cells, points, domain)
