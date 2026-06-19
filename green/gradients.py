from __future__ import annotations

import numpy as np

from sources import gradients_p1_tetra, locate_point_in_dolfinx_p1_tetra_mesh


def gradient_on_dolfinx_cell(poisson_solver, function, cell_id: int) -> np.ndarray:
    """Evaluate the constant gradient of a scalar P1 function on one cell."""
    V = poisson_solver.V
    cell_dofs = np.asarray(V.dofmap.cell_dofs(int(cell_id)), dtype=np.int64)
    if cell_dofs.shape != (4,):
        raise NotImplementedError("Green gradients currently support only scalar P1 tetra spaces")
    dof_coords = np.asarray(V.tabulate_dof_coordinates(), dtype=float)
    vertices = dof_coords[cell_dofs, :3]
    basis_gradients = gradients_p1_tetra(vertices)
    values = np.asarray(function.x.array, dtype=float)[cell_dofs]
    return np.asarray(values @ basis_gradients, dtype=float)


def gradients_at_candidate_cells(poisson_solver, functions, candidate_cell_ids) -> np.ndarray:
    """Evaluate Green gradients on DOLFINx candidate cell ids.

    Returns an array with shape ``(num_functions, num_candidates, 3)``.
    """
    functions = list(functions)
    cell_ids = np.asarray(candidate_cell_ids, dtype=np.int64)
    if cell_ids.ndim != 1:
        raise ValueError("candidate_cell_ids must be one-dimensional DOLFINx cell ids")
    gradients = np.empty((len(functions), cell_ids.size, 3), dtype=float)
    for function_index, function in enumerate(functions):
        for candidate_index, cell_id in enumerate(cell_ids):
            gradients[function_index, candidate_index] = gradient_on_dolfinx_cell(
                poisson_solver,
                function,
                int(cell_id),
            )
    return gradients


def locate_candidate_points_in_dolfinx(poisson_solver, candidate_points, tol: float = 1e-10) -> np.ndarray:
    """Locate candidate points and return cell ids in DOLFINx ordering."""
    points = np.asarray(candidate_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"candidate_points must have shape (n, 3), got {points.shape}")
    return np.asarray(
        [locate_point_in_dolfinx_p1_tetra_mesh(poisson_solver, point, tol=tol) for point in points],
        dtype=np.int64,
    )


def gradients_at_candidate_points(poisson_solver, functions, candidate_points, tol: float = 1e-10):
    """Locate candidate points and return ``(gradients, dolfinx_cell_ids)``."""
    cell_ids = locate_candidate_points_in_dolfinx(poisson_solver, candidate_points, tol=tol)
    return gradients_at_candidate_cells(poisson_solver, functions, cell_ids), cell_ids
