from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geometry import MeshData

from ._imports import require_fenicsx


@dataclass(frozen=True)
class DOLFINxP1Mapping:
    """Coordinate-based serial mapping between MeshData nodes and P1 dofs.

    ``node_to_dof[node_id]`` gives the corresponding scalar P1 DOLFINx dof.
    ``dof_to_node[dof_id]`` is the inverse permutation. The current MVP mapping
    is intentionally serial-only; distributed ownership needs a richer map.
    """

    node_to_dof: np.ndarray
    dof_to_node: np.ndarray
    is_serial: bool
    num_nodes: int
    num_dofs: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_to_dof = np.asarray(self.node_to_dof, dtype=np.int64)
        dof_to_node = np.asarray(self.dof_to_node, dtype=np.int64)
        if node_to_dof.ndim != 1:
            raise ValueError("node_to_dof must be one-dimensional")
        if dof_to_node.ndim != 1:
            raise ValueError("dof_to_node must be one-dimensional")
        if node_to_dof.shape != (int(self.num_nodes),):
            raise ValueError("node_to_dof length must match num_nodes")
        if dof_to_node.shape != (int(self.num_dofs),):
            raise ValueError("dof_to_node length must match num_dofs")
        if node_to_dof.size != dof_to_node.size:
            raise ValueError("serial P1 node/dof mapping requires equal node and dof counts")
        if np.any(node_to_dof < 0) or np.any(node_to_dof >= dof_to_node.size):
            raise ValueError("node_to_dof contains out-of-range dof ids")
        if np.any(dof_to_node < 0) or np.any(dof_to_node >= node_to_dof.size):
            raise ValueError("dof_to_node contains out-of-range node ids")
        if np.unique(node_to_dof).size != node_to_dof.size:
            raise ValueError("node_to_dof must be one-to-one")
        if np.unique(dof_to_node).size != dof_to_node.size:
            raise ValueError("dof_to_node must be one-to-one")
        if not np.array_equal(dof_to_node[node_to_dof], np.arange(node_to_dof.size, dtype=np.int64)):
            raise ValueError("dof_to_node must invert node_to_dof")
        object.__setattr__(self, "node_to_dof", node_to_dof)
        object.__setattr__(self, "dof_to_node", dof_to_node)
        object.__setattr__(self, "is_serial", bool(self.is_serial))
        object.__setattr__(self, "num_nodes", int(self.num_nodes))
        object.__setattr__(self, "num_dofs", int(self.num_dofs))
        object.__setattr__(self, "metadata", dict(self.metadata))


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


def _compute_node_to_dof_map_p1(solver, tol: float = 1e-12) -> np.ndarray:
    """Compute ``node_to_dof`` for scalar serial P1 spaces by coordinates."""
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


def build_p1_node_dof_mapping(solver, tol: float = 1e-12) -> DOLFINxP1Mapping:
    """Build a serial scalar P1 MeshData-node/DOLFINx-dof mapping object."""
    node_to_dof = _compute_node_to_dof_map_p1(solver, tol=tol)
    dof_to_node = np.empty_like(node_to_dof)
    dof_to_node[node_to_dof] = np.arange(node_to_dof.size, dtype=np.int64)
    comm = getattr(solver, "comm", None)
    return DOLFINxP1Mapping(
        node_to_dof=node_to_dof,
        dof_to_node=dof_to_node,
        is_serial=comm is None or int(comm.size) == 1,
        num_nodes=int(node_to_dof.size),
        num_dofs=int(dof_to_node.size),
        metadata={
            "ordering": "meshdata_node_to_dolfinx_dof",
            "tol": float(tol),
            "comm_size": None if comm is None else int(comm.size),
        },
    )


def build_node_to_dof_map_p1(solver, tol: float = 1e-12) -> np.ndarray:
    """Return cached ``node_to_dof`` for scalar P1 DOLFINx spaces.

    The returned permutation satisfies ``node_to_dof[node_id] == dof_id``.
    MeshData node ordering must never be assumed to equal DOLFINx ordering.
    This MVP helper supports serial scalar P1 spaces whose dofs are mesh
    vertices; distributed global-to-local mapping requires explicit ownership
    information and is intentionally rejected.
    """
    if hasattr(solver, "p1_node_dof_mapping"):
        return solver.p1_node_dof_mapping(tol=tol).node_to_dof
    return build_p1_node_dof_mapping(solver, tol=tol).node_to_dof
