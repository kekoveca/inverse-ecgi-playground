from __future__ import annotations

import numpy as np

from fem import get_p1_tetra_locator


def gradient_on_dolfinx_cell(poisson_solver, function, cell_id: int) -> np.ndarray:
    """Evaluate the constant gradient of a scalar P1 function on one cell."""
    locator = get_p1_tetra_locator(poisson_solver)
    cell_dofs_array, _ = locator.cell_geometry([int(cell_id)])
    cell_dofs = cell_dofs_array[0]
    basis_gradients = locator.basis_gradients([int(cell_id)])[0]
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
    if cell_ids.size == 0 or len(functions) == 0:
        return gradients
    locator = get_p1_tetra_locator(poisson_solver)
    cell_dofs, _ = locator.cell_geometry(cell_ids)
    basis_gradients = locator.basis_gradients(cell_ids)
    for function_index, function in enumerate(functions):
        values = np.asarray(function.x.array, dtype=float)[cell_dofs]
        gradients[function_index] = np.einsum("ca,cad->cd", values, basis_gradients)
    return gradients


def locate_candidate_points_in_dolfinx(poisson_solver, candidate_points, tol: float = 1e-10) -> np.ndarray:
    """Locate candidate points and return cell ids in DOLFINx ordering."""
    points = np.asarray(candidate_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"candidate_points must have shape (n, 3), got {points.shape}")
    locator = get_p1_tetra_locator(poisson_solver)
    return np.asarray(locator.locate_points(points, tol=tol), dtype=np.int64)


def gradients_at_candidate_points(poisson_solver, functions, candidate_points, tol: float = 1e-10):
    """Locate candidate points and return ``(gradients, dolfinx_cell_ids)``."""
    cell_ids = locate_candidate_points_in_dolfinx(poisson_solver, candidate_points, tol=tol)
    return gradients_at_candidate_cells(poisson_solver, functions, cell_ids), cell_ids
