from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._imports import require_fenicsx


@dataclass
class NeumannNullspaceHandler:
    """PETSc constant nullspace helper for pure Neumann scalar problems."""

    petsc_nullspace: object

    @classmethod
    def create(cls, comm=None) -> "NeumannNullspaceHandler":
        fx = require_fenicsx()
        MPI = fx["MPI"]
        PETSc = fx["PETSc"]
        if comm is None:
            comm = MPI.COMM_WORLD
        ns = PETSc.NullSpace().create(constant=True, comm=comm)
        return cls(petsc_nullspace=ns)

    def attach_to_matrix(self, A) -> None:
        A.setNullSpace(self.petsc_nullspace)

    def remove_from_vector(self, b) -> None:
        self.petsc_nullspace.remove(b)

    def test_matrix(self, A) -> bool:
        return bool(self.petsc_nullspace.test(A))

    def compatibility_residual(self, b) -> float | None:
        """Return the constant-nullspace RHS residual if it can be inspected."""
        if hasattr(b, "sum"):
            return float(b.sum())
        if hasattr(b, "array"):
            return float(np.asarray(b.array, dtype=float).sum())
        if hasattr(b, "getArray"):
            return float(np.asarray(b.getArray(), dtype=float).sum())
        return None

    def is_rhs_compatible(self, b, *, tol: float = 1e-10) -> bool | None:
        residual = self.compatibility_residual(b)
        if residual is None:
            return None
        return abs(residual) <= tol

    def check_rhs_compatible(self, b, *, tol: float = 1e-10) -> None:
        compatible = self.is_rhs_compatible(b, tol=tol)
        if compatible is False:
            residual = self.compatibility_residual(b)
            raise ValueError(f"RHS is incompatible with Neumann nullspace: constant residual={residual:g}")

    def fix_function_gauge(self, uh) -> None:
        """Set the solution gauge by subtracting its mean value."""
        values = uh.x.array
        if values.size == 0:
            return
        petsc_vec = getattr(uh.x, "petsc_vec", None)
        if petsc_vec is not None and hasattr(petsc_vec, "sum") and hasattr(petsc_vec, "getSize"):
            size = int(petsc_vec.getSize())
            mean_value = float(petsc_vec.sum()) / size if size else 0.0
        else:
            mean_value = float(np.asarray(values, dtype=float).mean())
        values[:] = values - mean_value
        uh.x.scatter_forward()


class ConstantNullspace(NeumannNullspaceHandler):
    """Backward-compatible name for ``NeumannNullspaceHandler``."""
