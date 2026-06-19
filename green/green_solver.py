from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .green_basis import GreenBasis, GreenSolveInfo
from .rhs import (
    check_measurement_matrix_compatibility,
    create_green_rhs_function,
    extract_measurement_rhs_row,
    measurement_matrix_row_sums,
)


class GreenSolver:
    """Solve reusable-stiffness Green problems ``K G_i = M_i^T``."""

    def __init__(
        self,
        poisson_solver,
        measurement_operator,
        keep_functions: bool = True,
        compatibility_tol: float = 1e-10,
    ) -> None:
        self.poisson_solver = poisson_solver
        self.measurement_operator = measurement_operator
        self.keep_functions = bool(keep_functions)
        self.compatibility_tol = float(compatibility_tol)

    def solve_one(self, row_index: int):
        """Solve one measurement-row Green problem and return function/info."""
        row = extract_measurement_rhs_row(self.measurement_operator, row_index)
        rhs_sum = float(row.sum())
        rhs_norm = float(np.linalg.norm(row))
        if abs(rhs_sum) > self.compatibility_tol * max(1.0, rhs_norm):
            raise ValueError(
                f"measurement row {row_index} is incompatible with the Neumann nullspace: sum={rhs_sum:.6g}"
            )
        rhs = create_green_rhs_function(
            self.poisson_solver,
            self.measurement_operator,
            row_index,
            tol=self.compatibility_tol,
        )
        function = self.poisson_solver.solve(rhs)
        diagnostics = getattr(self.poisson_solver, "diagnostics", None)
        reason = getattr(diagnostics, "converged_reason", None)
        residual = getattr(diagnostics, "residual_norm", None)
        info = GreenSolveInfo(
            row_index=int(row_index),
            rhs_sum=rhs_sum,
            rhs_norm=rhs_norm,
            converged=None if reason is None else bool(reason > 0),
            metadata={"converged_reason": reason, "residual_norm": residual},
        )
        return function, info

    def solve_all(self, row_indices=None) -> GreenBasis:
        """Solve all or selected compatible measurement rows with one matrix K."""
        if not check_measurement_matrix_compatibility(
            self.measurement_operator,
            tol=self.compatibility_tol,
        ):
            row_sums = measurement_matrix_row_sums(self.measurement_operator)
            worst = int(np.argmax(np.abs(row_sums)))
            raise ValueError(
                "measurement matrix is incompatible with the pure Neumann problem: "
                f"max row sum={row_sums[worst]:.6g} at row {worst}"
            )

        if row_indices is None:
            indices = np.arange(self.measurement_operator.num_electrodes, dtype=np.int64)
        else:
            indices = np.asarray(list(row_indices), dtype=np.int64)
            if indices.ndim != 1:
                raise ValueError("row_indices must be one-dimensional")
            if np.any(indices < 0) or np.any(indices >= self.measurement_operator.num_electrodes):
                raise IndexError("row_indices contain a measurement row outside the operator")

        functions = []
        solve_info = []
        for row_index in indices:
            function, info = self.solve_one(int(row_index))
            if self.keep_functions:
                functions.append(function)
            solve_info.append(asdict(info))

        return GreenBasis(
            measurement_operator=self.measurement_operator,
            functions=functions,
            reference=self.measurement_operator.reference,
            sign=1.0,
            metadata={
                "row_indices": indices.tolist(),
                "solve_info": solve_info,
                "sign_convention": "K G_i = M_i^T; transfer sign initially +1",
            },
        )
