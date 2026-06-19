from __future__ import annotations

import numpy as np

from geometry import ElectrodeSet
from fem import build_node_to_dof_map_p1
from measurements import MeasurementOperator, build_measurement_operator
from sources import PointDipole, assemble_point_dipole_rhs_petsc
from time import perf_counter

from .result import ForwardResult


def extract_nodal_values(function_or_vec) -> np.ndarray:
    """Extract nodal values from a DOLFINx Function-like object as a copy."""
    if hasattr(function_or_vec, "x") and hasattr(function_or_vec.x, "array"):
        return np.asarray(function_or_vec.x.array, dtype=float).copy()
    if hasattr(function_or_vec, "array"):
        return np.asarray(function_or_vec.array, dtype=float).copy()
    raise TypeError("expected a dolfinx.fem.Function-like object with x.array")


class ForwardSolver:
    """Compose dipole RHS assembly, Neumann solve and electrode measurements.

    The supplied Poisson solver owns the reusable stiffness matrix. If no
    electrodes/operator are provided, the result contains empty measurements.
    """

    def __init__(
        self,
        poisson_solver,
        electrodes: ElectrodeSet | None = None,
        measurement_operator: MeasurementOperator | None = None,
        reference: str = "average",
        reference_index: int | None = None,
        build_measurement_operator_if_needed: bool = True,
        measurement_sparse: bool = True,
        tol: float = 1e-10,
    ) -> None:
        self.poisson_solver = poisson_solver
        self.electrodes = electrodes
        self.reference = reference
        self.reference_index = reference_index
        self.measurement_sparse = bool(measurement_sparse)
        self.tol = float(tol)
        self._node_to_dof_map: np.ndarray | None = None

        if measurement_operator is None and electrodes is not None and build_measurement_operator_if_needed:
            mesh = self._mesh_data()
            measurement_operator = build_measurement_operator(
                mesh=mesh,
                electrodes=electrodes,
                reference=reference,
                reference_index=reference_index,
                sparse=measurement_sparse,
                tol=tol,
            )
        self.measurement_operator = measurement_operator

    def _mesh_data(self):
        if hasattr(self.poisson_solver, "mesh_data"):
            return self.poisson_solver.mesh_data
        if hasattr(self.poisson_solver, "input_mesh"):
            return self.poisson_solver.input_mesh
        raise TypeError("poisson_solver must expose mesh_data or input_mesh")

    def solve_potential(self, source: PointDipole):
        rhs = assemble_point_dipole_rhs_petsc(self.poisson_solver, source)
        return self.poisson_solver.solve(rhs)

    def _measurement_nodal_values(self, dof_values: np.ndarray) -> np.ndarray:
        """Return values in the ordering expected by the measurement operator."""
        if self.measurement_operator is None:
            return dof_values
        ordering = self.measurement_operator.metadata.get("ordering", "meshdata_node")
        if ordering == "dolfinx_dof":
            return dof_values
        if ordering != "meshdata_node":
            raise ValueError(f"unsupported MeasurementOperator ordering {ordering!r}")
        if self._node_to_dof_map is None:
            self._node_to_dof_map = build_node_to_dof_map_p1(self.poisson_solver, tol=self.tol)
        return dof_values[self._node_to_dof_map]

    def solve(self, source: PointDipole) -> ForwardResult:
        t_start = perf_counter()
        
        potential = self.solve_potential(source)
        nodal_values = extract_nodal_values(potential)

        if self.measurement_operator is None:
            raw_measurements = np.empty((0,), dtype=float)
            measurements = np.empty((0,), dtype=float)
        else:
            measurement_values = self._measurement_nodal_values(nodal_values)
            raw_measurements = self.measurement_operator.evaluate_raw(measurement_values)
            measurements = self.measurement_operator.evaluate(measurement_values)

        t = perf_counter() - t_start
        metadata = {
            "solver": self.poisson_solver.__class__.__name__,
            "has_measurement_operator": self.measurement_operator is not None,
            "time": t,
        }
        return ForwardResult(
            source=source,
            potential=potential,
            nodal_values=nodal_values,
            raw_measurements=raw_measurements,
            measurements=measurements,
            reference=self.reference,
            metadata=metadata,
        )
