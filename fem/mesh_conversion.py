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
    return dmesh.create_mesh(comm, cells, domain, points)


def build_node_to_dof_map_p1(solver, tol: float = 1e-12) -> np.ndarray:
    """Match MeshData node ids to scalar P1 DOLFINx dof ids by coordinates.

    The returned permutation satisfies ``node_to_dof[node_id] == dof_id``.
    MeshData node ordering must never be assumed to equal DOLFINx ordering.
    This MVP helper supports serial scalar P1 spaces whose dofs are mesh
    vertices; distributed global-to-local mapping requires explicit ownership
    information and is intentionally rejected.
    """
    mesh_data = getattr(solver, "mesh_data", None)
    V = getattr(solver, "V", None)
    if not isinstance(mesh_data, MeshData) or V is None:
        raise TypeError("solver must expose MeshData as mesh_data and a DOLFINx FunctionSpace as V")
    if mesh_data.cell_type != "tetra" or mesh_data.cells.shape[1] != 4:
        raise NotImplementedError("node-to-dof mapping currently supports scalar P1 tetra spaces")
    if int(getattr(solver, "degree", 1)) != 1:
        raise NotImplementedError("node-to-dof mapping currently supports scalar P1 tetra spaces")

    comm = getattr(solver, "comm", None)
    if comm is not None and int(comm.size) != 1:
        raise NotImplementedError("node-to-dof coordinate mapping currently supports serial DOLFINx meshes only")

    node_coords = np.asarray(mesh_data.points, dtype=float)
    dof_coords = np.asarray(V.tabulate_dof_coordinates(), dtype=float)[:, : node_coords.shape[1]]
    if node_coords.shape[0] != dof_coords.shape[0]:
        raise ValueError(
            "scalar P1 node-to-dof mapping requires one DOLFINx dof per MeshData node; "
            f"got {node_coords.shape[0]} nodes and {dof_coords.shape[0]} dofs"
        )

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - scipy is a project dependency
        raise ImportError("scipy is required to match MeshData nodes to DOLFINx P1 dofs") from exc

    distances, node_to_dof = cKDTree(dof_coords).query(node_coords, k=1)
    scale = max(1.0, float(np.ptp(node_coords, axis=0).max(initial=0.0)))
    threshold = float(tol) * scale
    if np.any(distances > threshold):
        node_id = int(np.argmax(distances))
        raise ValueError(
            "could not match every MeshData node to a DOLFINx dof within tolerance: "
            f"node {node_id}, distance={distances[node_id]:.6g}, tolerance={threshold:.6g}"
        )
    node_to_dof = np.asarray(node_to_dof, dtype=np.int64)
    if np.unique(node_to_dof).size != node_to_dof.size:
        raise ValueError("MeshData node-to-DOLFINx dof matching is not one-to-one")
    return node_to_dof
