from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from geometry import MeshData


def _solver_function_space(solver):
    V = getattr(solver, "V", None)
    if V is None:
        raise TypeError("solver must expose a DOLFINx FunctionSpace as V")
    return V


def _num_local_cells(solver) -> int:
    domain = getattr(solver, "domain", None)
    if domain is None:
        raise TypeError("solver must expose a DOLFINx mesh as domain")
    tdim = int(domain.topology.dim)
    index_map = domain.topology.index_map(tdim)
    if index_map is None:
        raise RuntimeError("DOLFINx cell index map is not available")
    return int(index_map.size_local)


def _validate_p1_tetra_solver(solver) -> None:
    mesh_data = getattr(solver, "mesh_data", None)
    if not isinstance(mesh_data, MeshData):
        raise TypeError("solver must expose MeshData as mesh_data")
    if mesh_data.cell_type != "tetra" or mesh_data.cells.shape[1] != 4:
        raise NotImplementedError("DOLFINx P1 tetra locator supports only tetra MeshData")
    if int(getattr(solver, "degree", 1)) != 1:
        raise NotImplementedError("DOLFINx P1 tetra locator supports only scalar P1 spaces")


def _barycentric_coordinates_batch(point: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Return barycentric coordinates for one point against many tetrahedra."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 3 or vertices.shape[1:] != (4, 3):
        raise ValueError(f"vertices must have shape (n_cells, 4, 3), got {vertices.shape}")
    matrices = np.empty((vertices.shape[0], 4, 4), dtype=float)
    matrices[:, :3, :] = np.transpose(vertices, (0, 2, 1))
    matrices[:, 3, :] = 1.0
    rhs = np.broadcast_to(np.array([point[0], point[1], point[2], 1.0], dtype=float), (vertices.shape[0], 4))
    return np.linalg.solve(matrices, rhs)


def _basis_gradients_batch(vertices: np.ndarray) -> np.ndarray:
    """Return P1 basis gradients for many tetrahedra in local dof order."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 3 or vertices.shape[1:] != (4, 3):
        raise ValueError(f"vertices must have shape (n_cells, 4, 3), got {vertices.shape}")
    matrices = np.empty((vertices.shape[0], 4, 4), dtype=float)
    matrices[:, :3, :] = np.transpose(vertices, (0, 2, 1))
    matrices[:, 3, :] = 1.0
    return np.linalg.inv(matrices)[:, :, :3].copy()


@dataclass
class DOLFINxP1TetraLocator:
    """Cached local-cell locator for scalar P1 tetrahedral DOLFINx meshes.

    Cell ids returned by this object are local DOLFINx cell ids, not MeshData
    cell ids. The locator is intentionally local-cell only, matching the
    project's current serial/local DOLFINx assumptions.
    """

    solver: Any
    dof_coords: np.ndarray
    cell_dofs: np.ndarray
    cell_vertices: np.ndarray
    cell_centers: np.ndarray
    tree: cKDTree
    metadata: dict[str, Any]

    @classmethod
    def from_solver(cls, solver) -> "DOLFINxP1TetraLocator":
        _validate_p1_tetra_solver(solver)
        V = _solver_function_space(solver)
        num_cells = _num_local_cells(solver)
        if num_cells <= 0:
            raise ValueError("DOLFINx mesh has no local cells")

        dof_coords = np.asarray(V.tabulate_dof_coordinates(), dtype=float)[:, :3]
        cell_dofs = np.empty((num_cells, 4), dtype=np.int64)
        for cell_id in range(num_cells):
            dofs = np.asarray(V.dofmap.cell_dofs(cell_id), dtype=np.int64)
            if dofs.shape != (4,):
                raise NotImplementedError("DOLFINx P1 tetra locator supports only scalar P1 tetra spaces")
            cell_dofs[cell_id] = dofs

        cell_vertices = dof_coords[cell_dofs]
        cell_centers = cell_vertices.mean(axis=1)
        return cls(
            solver=solver,
            dof_coords=dof_coords,
            cell_dofs=cell_dofs,
            cell_vertices=cell_vertices,
            cell_centers=cell_centers,
            tree=cKDTree(cell_centers),
            metadata={
                "num_local_cells": int(num_cells),
                "num_dofs": int(dof_coords.shape[0]),
                "ordering": "dolfinx_local_cell_id",
            },
        )

    def _validated_cell_ids(self, candidate_cell_ids=None) -> np.ndarray:
        num_cells = self.cell_dofs.shape[0]
        if candidate_cell_ids is None:
            return np.arange(num_cells, dtype=np.int64)
        cell_ids = np.asarray(candidate_cell_ids, dtype=np.int64)
        if cell_ids.ndim != 1:
            raise ValueError("candidate_cell_ids must be one-dimensional")
        if cell_ids.size == 0:
            raise ValueError("candidate_cell_ids must contain at least one cell")
        if cell_ids.min() < 0 or cell_ids.max() >= num_cells:
            raise ValueError("candidate_cell_ids contain ids outside local DOLFINx cells")
        return cell_ids

    def locate_point(self, point, candidate_cell_ids=None, tol: float = 1e-10, initial_k: int = 8) -> int:
        """Locate one point and return its local DOLFINx cell id."""
        return int(
            self.locate_points(
                np.asarray(point, dtype=float).reshape(1, 3),
                candidate_cell_ids=candidate_cell_ids,
                tol=tol,
                initial_k=initial_k,
            )[0]
        )

    def locate_points(
        self,
        points,
        candidate_cell_ids=None,
        tol: float = 1e-10,
        initial_k: int = 8,
        return_barycentric: bool = False,
    ):
        """Locate points in local DOLFINx cell ordering.

        A centroid KD-tree orders candidate cells; barycentric coordinates are
        still used as the acceptance test, so nearest-centroid cells are never
        accepted without a geometric containment check.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (n_points, 3), got {points.shape}")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must contain only finite values")
        if initial_k < 1:
            raise ValueError("initial_k must be positive")
        if points.shape[0] == 0:
            empty_ids = np.empty((0,), dtype=np.int64)
            if return_barycentric:
                return empty_ids, np.empty((0, 4), dtype=float)
            return empty_ids

        cell_ids = self._validated_cell_ids(candidate_cell_ids)
        if candidate_cell_ids is None:
            tree = self.tree
        else:
            tree = cKDTree(self.cell_centers[cell_ids])

        located = np.full(points.shape[0], -1, dtype=np.int64)
        barycentric = np.full((points.shape[0], 4), np.nan, dtype=float)
        k = min(int(initial_k), cell_ids.size)

        while np.any(located < 0):
            unresolved = np.flatnonzero(located < 0)
            _, local_indices = tree.query(points[unresolved], k=k)
            if k == 1:
                local_indices = local_indices[:, np.newaxis]

            for row, point_id in enumerate(unresolved):
                point = points[point_id]
                queried_cell_ids = cell_ids[np.asarray(local_indices[row], dtype=np.int64)]
                lambdas = _barycentric_coordinates_batch(point, self.cell_vertices[queried_cell_ids])
                inside = (
                    np.all(lambdas >= -tol, axis=1)
                    & np.all(lambdas <= 1.0 + tol, axis=1)
                    & (np.abs(lambdas.sum(axis=1) - 1.0) <= tol)
                )
                if np.any(inside):
                    local_hit = int(np.flatnonzero(inside)[0])
                    located[point_id] = int(queried_cell_ids[local_hit])
                    barycentric[point_id] = lambdas[local_hit]

            if not np.any(located < 0):
                break
            if k == cell_ids.size:
                first_missing = int(np.flatnonzero(located < 0)[0])
                raise ValueError(f"point {first_missing} is not inside any local DOLFINx P1 tetra cell")
            k = min(2 * k, cell_ids.size)

        if return_barycentric:
            return located, barycentric
        return located

    def cell_geometry(self, cell_ids) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(cell_dofs, vertices)`` for local DOLFINx cell ids."""
        ids = self._validated_cell_ids(np.asarray(cell_ids, dtype=np.int64).reshape(-1))
        return self.cell_dofs[ids].copy(), self.cell_vertices[ids].copy()

    def basis_gradients(self, cell_ids) -> np.ndarray:
        """Return P1 basis gradients for local DOLFINx cells."""
        ids = self._validated_cell_ids(np.asarray(cell_ids, dtype=np.int64).reshape(-1))
        return _basis_gradients_batch(self.cell_vertices[ids])


def get_p1_tetra_locator(solver) -> DOLFINxP1TetraLocator:
    """Return a cached DOLFINx P1 tetra locator for ``solver``."""
    if hasattr(solver, "p1_tetra_locator"):
        return solver.p1_tetra_locator()
    locator = getattr(solver, "_p1_tetra_locator", None)
    if locator is None:
        locator = DOLFINxP1TetraLocator.from_solver(solver)
        setattr(solver, "_p1_tetra_locator", locator)
    return locator
