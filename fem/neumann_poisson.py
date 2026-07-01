from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any

import numpy as np

from geometry import MeshData, TorsoGeometry

from ._imports import require_fenicsx
from .mesh_conversion import create_dolfinx_mesh
from .nullspace import NeumannNullspaceHandler

KSPType = Literal["cg", "gmres", "preonly"]
PCType = Literal["hypre", "gamg", "jacobi", "lu", "none"]


@dataclass(frozen=True)
class SolverDiagnostics:
    ksp_type: str
    pc_type: str
    converged_reason: int | None
    residual_norm: float | None
    nullspace_test_passed: bool | None


class FunctionSpaceFactory:
    """Create the DOLFINx mesh and scalar P1 Lagrange function space."""

    def __init__(self, *, degree: int = 1) -> None:
        self.degree = int(degree)
        if self.degree != 1:
            raise ValueError("FEM MVP supports only P1 Lagrange elements")

    def create(self, mesh: MeshData, *, comm: Any | None = None, fx: dict[str, Any] | None = None):
        fx = fx or require_fenicsx()
        MPI = fx["MPI"]
        fem = fx["fem"]
        if comm is None:
            comm = MPI.COMM_WORLD
        domain = create_dolfinx_mesh(mesh, comm=comm)
        V = fem.functionspace(domain, ("Lagrange", self.degree))
        return domain, V


class StiffnessOperator:
    """Assemble and own the FEM stiffness matrix ``K`` for one geometry."""

    def __init__(self, *, V, domain, sigma: float = 1.0, fx: dict[str, Any] | None = None) -> None:
        self._fx = fx or require_fenicsx()
        self.V = V
        self.domain = domain
        self.sigma = float(sigma)
        self.a_form = None
        self.K = None
        self.assemble()

    def assemble(self) -> None:
        fem = self._fx["fem"]
        fem_petsc = self._fx["fem_petsc"]
        ufl = self._fx["ufl"]
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        dx = ufl.dx(domain=self.domain)
        self.a_form = fem.form(self.sigma * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx)
        self.K = fem_petsc.assemble_matrix(self.a_form, bcs=[])
        self.K.assemble()

    def destroy(self) -> None:
        if self.K is not None:
            self.K.destroy()
            self.K = None


class LinearSolver:
    """PETSc KSP wrapper that reuses one stiffness matrix for many RHS values."""

    def __init__(
        self,
        *,
        K,
        comm,
        fx: dict[str, Any],
        ksp_type: KSPType = "cg",
        pc_type: PCType = "hypre",
    ) -> None:
        self.K = K
        self.comm = comm
        self._fx = fx
        self.PETSc = fx["PETSc"]
        self.ksp_type = ksp_type
        self.pc_type = pc_type
        self.ksp = None
        self.setup()

    def setup(self) -> None:
        ksp = self.PETSc.KSP().create(self.comm)
        ksp.setOperators(self.K)
        ksp.setType(self.ksp_type)
        pc = ksp.getPC()
        if self.pc_type == "none":
            pc.setType(self.PETSc.PC.Type.NONE)
        else:
            pc.setType(self.pc_type)
        ksp.setFromOptions()
        self.ksp = ksp

    def solve(self, b, x) -> SolverDiagnostics:
        self.ksp.solve(b, x)
        return SolverDiagnostics(
            ksp_type=self.ksp_type,
            pc_type=self.pc_type,
            converged_reason=int(self.ksp.getConvergedReason()),
            residual_norm=float(self.ksp.getResidualNorm()),
            nullspace_test_passed=None,
        )

    def destroy(self) -> None:
        if self.ksp is not None:
            self.ksp.destroy()
            self.ksp = None


class FEMProblem:
    """FEniCSx-backed core for scalar Poisson problems with Neumann nullspace.

    The problem owns the main artifact ``K``: a FEM stiffness matrix assembled
    once for one geometry. The same matrix and KSP setup are then reused for
    repeated solves with different right-hand sides.

    The class assembles the stiffness matrix

        K_ij = ∫ sigma ∇phi_i · ∇phi_j dx

    with scalar constant conductivity ``sigma`` and P1 Lagrange elements.
    """

    def __init__(
        self,
        mesh: MeshData | TorsoGeometry,
        *,
        degree: int = 1,
        sigma: float = 1.0,
        comm: Any | None = None,
        ksp_type: KSPType = "cg",
        pc_type: PCType = "hypre",
        test_nullspace: bool = True,
        rhs_compatibility_tol: float = 1e-10,
    ) -> None:
        fx = require_fenicsx()
        self._fx = fx
        self.MPI = fx["MPI"]
        self.PETSc = fx["PETSc"]
        self.fem = fx["fem"]
        self.fem_petsc = fx["fem_petsc"]
        self.ufl = fx["ufl"]

        self.geometry = mesh if isinstance(mesh, TorsoGeometry) else None
        self.surface_mesh = mesh.surface_mesh if isinstance(mesh, TorsoGeometry) else None
        if isinstance(mesh, TorsoGeometry):
            mesh_data = mesh.volume_mesh
        else:
            mesh_data = mesh
        if mesh_data.cell_type not in ("triangle", "tetra"):
            raise ValueError("NeumannPoissonSolver requires a triangle or tetra volume mesh")
        if mesh_data.num_cells == 0:
            raise ValueError("NeumannPoissonSolver requires a mesh with at least one cell")

        self.mesh_data = mesh_data
        self.comm = comm if comm is not None else self.MPI.COMM_WORLD
        self.degree = int(degree)
        if self.degree != 1:
            raise ValueError("FEM MVP supports only P1 Lagrange elements")
        self.sigma = float(sigma)
        self.ksp_type = ksp_type
        self.pc_type = pc_type
        self.rhs_compatibility_tol = float(rhs_compatibility_tol)

        self.function_space_factory = FunctionSpaceFactory(degree=self.degree)
        self.domain, self.V = self.function_space_factory.create(mesh_data, comm=self.comm, fx=fx)
        self._p1_node_dof_mapping = None
        self._p1_node_dof_mapping_tol: float | None = None
        self._p1_tetra_locator = None

        self.a_form = None
        self.A = None
        self.K = None
        self.stiffness_operator: StiffnessOperator | None = None
        self.nullspace: NeumannNullspaceHandler | None = None
        self.nullspace_handler: NeumannNullspaceHandler | None = None
        self.linear_solver: LinearSolver | None = None
        self.ksp = None
        self.diagnostics = SolverDiagnostics(
            ksp_type=ksp_type,
            pc_type=pc_type,
            converged_reason=None,
            residual_norm=None,
            nullspace_test_passed=None,
        )

        self._assemble_stiffness()
        self._setup_nullspace(test_nullspace=test_nullspace)
        self._setup_ksp()

    def _assemble_stiffness(self) -> None:
        self.stiffness_operator = StiffnessOperator(
            V=self.V,
            domain=self.domain,
            sigma=self.sigma,
            fx=self._fx,
        )
        self.a_form = self.stiffness_operator.a_form
        self.K = self.stiffness_operator.K
        self.A = self.K

    def _setup_nullspace(self, *, test_nullspace: bool) -> None:
        self.nullspace = NeumannNullspaceHandler.create(comm=self.comm)
        self.nullspace_handler = self.nullspace
        self.nullspace.attach_to_matrix(self.A)
        passed = None
        if test_nullspace:
            passed = self.nullspace.test_matrix(self.A)
            if not passed:
                raise RuntimeError("Constant nullspace test failed for assembled Neumann matrix")
        self.diagnostics = SolverDiagnostics(
            ksp_type=self.ksp_type,
            pc_type=self.pc_type,
            converged_reason=None,
            residual_norm=None,
            nullspace_test_passed=passed,
        )

    def _setup_ksp(self) -> None:
        self.linear_solver = LinearSolver(
            K=self.K,
            comm=self.comm,
            fx=self._fx,
            ksp_type=self.ksp_type,
            pc_type=self.pc_type,
        )
        self.ksp = self.linear_solver.ksp

    def p1_node_dof_mapping(self, *, tol: float = 1e-12):
        """Return cached MeshData-node/DOLFINx-dof mapping for serial P1 spaces.

        The mapping is explicitly in ``MeshData node id -> DOLFINx dof id``
        order. It is serial-only in the MVP and raises for distributed meshes.
        """
        tol = float(tol)
        if (
            self._p1_node_dof_mapping is None
            or self._p1_node_dof_mapping_tol is None
            or tol < self._p1_node_dof_mapping_tol
        ):
            from .mesh_conversion import build_p1_node_dof_mapping

            self._p1_node_dof_mapping = build_p1_node_dof_mapping(self, tol=tol)
            self._p1_node_dof_mapping_tol = tol
        return self._p1_node_dof_mapping

    def p1_tetra_locator(self):
        """Return cached local-cell locator for scalar P1 tetra DOLFINx meshes."""
        if self._p1_tetra_locator is None:
            from .p1_locator import DOLFINxP1TetraLocator

            self._p1_tetra_locator = DOLFINxP1TetraLocator.from_solver(self)
        return self._p1_tetra_locator

    @property
    def node_to_dof_map(self) -> np.ndarray:
        """Cached ``node_to_dof[node_id] == dof_id`` map for scalar P1."""
        return self.p1_node_dof_mapping().node_to_dof

    @property
    def dof_to_node_map(self) -> np.ndarray:
        """Cached inverse ``dof_to_node[dof_id] == node_id`` map for scalar P1."""
        return self.p1_node_dof_mapping().dof_to_node

    def zero_function(self):
        """Return a zero ``dolfinx.fem.Function`` in the solver space."""
        uh = self.fem.Function(self.V)
        uh.x.array[:] = 0.0
        uh.x.scatter_forward()
        return uh

    def rhs_from_local_array(self, values: np.ndarray):
        """Create a RHS Function from a local vector array.

        This is mainly a convenience for tests and prototypes. Production RHS
        assemblers should fill a DOLFINx Function/PETSc Vec directly using the
        correct dof ownership rules.
        """
        b = self.zero_function()
        values = np.asarray(values, dtype=float)
        if values.shape != b.x.array.shape:
            raise ValueError(f"values must have local shape {b.x.array.shape}, got {values.shape}")
        b.x.array[:] = values
        b.x.scatter_forward()
        return b

    def _as_petsc_vec(self, rhs):
        if hasattr(rhs, "x") and hasattr(rhs.x, "petsc_vec"):
            return rhs.x.petsc_vec
        if hasattr(rhs, "petsc_vec"):
            return rhs.petsc_vec
        # Accept a raw PETSc Vec.
        if rhs.__class__.__name__ == "Vec":
            return rhs
        raise TypeError("rhs must be a dolfinx.fem.Function, a la.Vector-like object, or a PETSc Vec")

    def check_rhs_compatible(self, rhs, *, tol: float | None = None) -> None:
        if self.nullspace_handler is None:
            raise RuntimeError("Nullspace handler is not initialized")
        self.nullspace_handler.check_rhs_compatible(
            self._as_petsc_vec(rhs),
            tol=self.rhs_compatibility_tol if tol is None else tol,
        )

    def solve(
        self,
        rhs,
        *,
        remove_nullspace_component: bool = True,
        check_rhs_compatibility: bool = True,
        check_convergence: bool = True,
        fix_gauge: bool = True,
    ):
        """Solve ``K u = rhs`` and return a DOLFINx Function ``u``.

        For a pure Neumann problem the RHS must be compatible with the constant
        nullspace. By default the constant component is removed before solving.
        """
        if self.linear_solver is None or self.ksp is None or self.nullspace is None:
            raise RuntimeError("Solver is not initialized")

        b = self._as_petsc_vec(rhs).copy()
        try:
            if remove_nullspace_component:
                self.nullspace.remove_from_vector(b)
            if check_rhs_compatibility:
                self.nullspace.check_rhs_compatible(b, tol=self.rhs_compatibility_tol)

            uh = self.zero_function()
            x = uh.x.petsc_vec
            x.set(0.0)
            solve_diagnostics = self.linear_solver.solve(b, x)
            uh.x.scatter_forward()
            if fix_gauge:
                self.nullspace.fix_function_gauge(uh)

            self.diagnostics = SolverDiagnostics(
                ksp_type=self.ksp_type,
                pc_type=self.pc_type,
                converged_reason=solve_diagnostics.converged_reason,
                residual_norm=solve_diagnostics.residual_norm,
                nullspace_test_passed=self.diagnostics.nullspace_test_passed,
            )

            if check_convergence and self.diagnostics.converged_reason <= 0:
                raise RuntimeError(
                    f"PETSc KSP did not converge, reason={self.diagnostics.converged_reason}, "
                    f"residual={self.diagnostics.residual_norm}"
                )
            return uh
        finally:
            b.destroy()

    def destroy(self) -> None:
        """Collectively destroy PETSc objects owned by the solver."""
        self._p1_tetra_locator = None
        if self.linear_solver is not None:
            self.linear_solver.destroy()
            self.linear_solver = None
            self.ksp = None
        if self.stiffness_operator is not None:
            self.stiffness_operator.destroy()
            self.stiffness_operator = None
            self.A = None
            self.K = None


class NeumannPoissonSolver(FEMProblem):
    """P1 DOLFINx solver for scalar Poisson problems with pure Neumann BCs.

    The solver owns a reusable stiffness matrix/KSP and handles the constant
    PETSc nullspace. MeshData node/cell ids are not DOLFINx dof/cell ids.
    This class is the public, backward-compatible name for ``FEMProblem``.
    """
