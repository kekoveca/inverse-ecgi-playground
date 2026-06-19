from __future__ import annotations

import numpy as np

from fem import build_node_to_dof_map_p1 as _build_node_to_dof_map_p1


def get_measurement_matrix(measurement_operator):
    """Return the two-dimensional measurement matrix ``M = R @ P``."""
    matrix = measurement_operator.matrix()
    if getattr(matrix, "ndim", 2) != 2 or len(matrix.shape) != 2:
        raise ValueError("measurement operator matrix must be two-dimensional")
    return matrix


def measurement_matrix_row_sums(measurement_operator) -> np.ndarray:
    """Return sums of rows of the measurement matrix."""
    matrix = get_measurement_matrix(measurement_operator)
    return np.asarray(matrix.sum(axis=1), dtype=float).reshape(-1)


def check_measurement_matrix_compatibility(measurement_operator, tol: float = 1e-10) -> bool:
    """Check that every measurement row is compatible with constant nullspace."""
    matrix = get_measurement_matrix(measurement_operator)
    row_sums = measurement_matrix_row_sums(measurement_operator)
    if hasattr(matrix, "multiply"):
        row_norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1), dtype=float).reshape(-1))
    else:
        row_norms = np.linalg.norm(np.asarray(matrix, dtype=float), axis=1)
    return bool(np.all(np.abs(row_sums) <= float(tol) * np.maximum(1.0, row_norms)))


def extract_measurement_rhs_row(measurement_operator, row_index: int, dense: bool = True) -> np.ndarray:
    """Extract one row of ``M`` as a MeshData-node-ordered vector."""
    matrix = get_measurement_matrix(measurement_operator)
    row_index = int(row_index)
    if row_index < 0 or row_index >= matrix.shape[0]:
        raise IndexError(f"measurement row {row_index} is outside [0, {matrix.shape[0]})")
    row = matrix.getrow(row_index) if hasattr(matrix, "getrow") else matrix[row_index]
    if not dense:
        return row
    if hasattr(row, "toarray"):
        row = row.toarray()
    return np.asarray(row, dtype=float).reshape(-1)


def build_node_to_dof_map_p1(poisson_solver, tol: float = 1e-12) -> np.ndarray:
    """Return the MeshData-node to scalar-P1-DOLFINx-dof permutation."""
    return _build_node_to_dof_map_p1(poisson_solver, tol=tol)


def create_function_from_meshdata_nodal_values(
    poisson_solver,
    nodal_values,
    name: str | None = None,
    tol: float = 1e-12,
):
    """Create a DOLFINx Function from values in MeshData node ordering.

    Values are mapped by coordinates. Directly copying a MeshData vector into
    ``Function.x.array`` is invalid because node and dof permutations may differ.
    """
    values = np.asarray(nodal_values, dtype=float)
    num_nodes = int(poisson_solver.mesh_data.num_points)
    if values.shape != (num_nodes,):
        raise ValueError(f"nodal_values must have shape ({num_nodes},), got {values.shape}")
    node_to_dof = build_node_to_dof_map_p1(poisson_solver, tol=tol)
    function = poisson_solver.zero_function()
    function.x.array[:] = 0.0
    function.x.array[node_to_dof] = values
    function.x.scatter_forward()
    if name is not None:
        function.name = str(name)
    return function


def create_green_rhs_function(poisson_solver, measurement_operator, row_index: int, tol: float = 1e-12):
    """Create ``M_i^T`` in DOLFINx DOF ordering for one Green solve."""
    row = extract_measurement_rhs_row(measurement_operator, row_index, dense=True)
    return create_function_from_meshdata_nodal_values(
        poisson_solver,
        row,
        name=f"green_rhs_{int(row_index)}",
        tol=tol,
    )
