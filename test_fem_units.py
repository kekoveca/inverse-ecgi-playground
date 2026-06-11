import types

import numpy as np
import pytest

import fem
from fem._imports import require_fenicsx
from fem.mesh_conversion import create_dolfinx_mesh, infer_cell_type
import fem.mesh_conversion as mesh_conversion
import fem.neumann_poisson as neumann_poisson
import fem.nullspace as nullspace_module
from fem.neumann_poisson import (
    FEMProblem,
    FunctionSpaceFactory,
    LinearSolver,
    NeumannPoissonSolver,
    SolverDiagnostics,
    StiffnessOperator,
)
from fem.nullspace import ConstantNullspace, NeumannNullspaceHandler
from geometry import ElectrodeSet, MeshData, SourceRegion, TorsoGeometry


def single_tetra_mesh(name="single_tet"):
    return MeshData(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
        cell_type="tetra",
        name=name,
    )


def single_triangle_mesh(points=None, name="single_tri"):
    if points is None:
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    return MeshData(
        points=np.asarray(points, dtype=float),
        cells=np.array([[0, 1, 2]], dtype=np.int64),
        cell_type="triangle",
        name=name,
    )


def test_public_fem_exports_are_available():
    for name in fem.__all__:
        assert hasattr(fem, name), name
    assert fem.FEMProblem is FEMProblem
    assert fem.FunctionSpaceFactory is FunctionSpaceFactory
    assert fem.StiffnessOperator is StiffnessOperator
    assert fem.NeumannNullspaceHandler is NeumannNullspaceHandler
    assert fem.LinearSolver is LinearSolver
    assert fem.NeumannPoissonSolver is NeumannPoissonSolver
    assert fem.SolverDiagnostics is SolverDiagnostics
    assert fem.ConstantNullspace is ConstantNullspace


def test_require_fenicsx_reports_missing_stack_clearly(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "mpi4py":
            raise ImportError("missing mpi4py")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError, match="requires FEniCSx"):
        require_fenicsx()


def test_infer_cell_type_supports_triangle_and_tetra_simplexes():
    assert infer_cell_type(single_tetra_mesh()) == "tetrahedron"
    assert infer_cell_type(single_triangle_mesh()) == "triangle"
    assert infer_cell_type(single_triangle_mesh(points=[[0, 0, 0], [1, 0, 0], [0, 1, 1]])) == "triangle"


def test_infer_cell_type_rejects_bad_tetra_dimension_and_unsupported_cells():
    tetra_2d = MeshData(
        points=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
        cell_type="tetra",
    )
    line = MeshData(
        points=np.array([[0.0, 0.0], [1.0, 0.0]]),
        cells=np.array([[0, 1]], dtype=np.int64),
        cell_type="line",
    )

    with pytest.raises(ValueError, match="tetra mesh must have 3D"):
        infer_cell_type(tetra_2d)
    with pytest.raises(ValueError, match="Unsupported FEM cell_type"):
        infer_cell_type(line)


def test_create_dolfinx_mesh_uses_fenicsx_factories_and_default_comm(monkeypatch):
    calls = {}

    class FakeBasixUfl:
        @staticmethod
        def element(family, cell_name, degree, shape):
            calls["element"] = (family, cell_name, degree, shape)
            return {"family": family, "cell_name": cell_name, "degree": degree, "shape": shape}

    class FakeUfl:
        @staticmethod
        def Mesh(coordinate_element):
            calls["domain"] = coordinate_element
            return {"domain": coordinate_element}

    class FakeDmesh:
        @staticmethod
        def create_mesh(comm, cells, domain, points):
            calls["create_mesh"] = (comm, cells.copy(), domain, points.copy())
            return "dolfinx-mesh"

    fake_fx = {
        "MPI": types.SimpleNamespace(COMM_WORLD="WORLD"),
        "basix_ufl": FakeBasixUfl,
        "dmesh": FakeDmesh,
        "ufl": FakeUfl,
    }
    monkeypatch.setattr(mesh_conversion, "require_fenicsx", lambda: fake_fx)

    mesh = single_tetra_mesh()
    created = create_dolfinx_mesh(mesh)

    assert created == "dolfinx-mesh"
    assert calls["element"] == ("Lagrange", "tetrahedron", 1, (3,))
    comm, cells, domain, points = calls["create_mesh"]
    assert comm == "WORLD"
    assert np.array_equal(cells, mesh.cells)
    assert np.allclose(points, mesh.points)
    assert domain == {"domain": calls["domain"]}


def test_create_dolfinx_mesh_rejects_zero_cell_mesh_after_lazy_import(monkeypatch):
    monkeypatch.setattr(
        mesh_conversion,
        "require_fenicsx",
        lambda: {
            "MPI": types.SimpleNamespace(COMM_WORLD="WORLD"),
            "basix_ufl": object(),
            "dmesh": object(),
            "ufl": object(),
        },
    )
    empty = MeshData(points=np.zeros((3, 2)), cells=np.empty((0, 3), dtype=np.int64), cell_type="triangle")

    with pytest.raises(ValueError, match="zero cells"):
        create_dolfinx_mesh(empty)


def test_constant_nullspace_create_and_delegates_to_petsc(monkeypatch):
    calls = {}

    class FakePetscNullSpace:
        def create(self, *, constant, comm):
            calls["create"] = (constant, comm)
            return self

        def remove(self, b):
            calls["remove"] = b

        def test(self, A):
            calls["test"] = A
            return 1

    fake_ns = FakePetscNullSpace()
    fake_petsc = types.SimpleNamespace(NullSpace=lambda: fake_ns)
    monkeypatch.setattr(
        nullspace_module,
        "require_fenicsx",
        lambda: {"MPI": types.SimpleNamespace(COMM_WORLD="WORLD"), "PETSc": fake_petsc},
    )

    ns = ConstantNullspace.create()
    matrix = types.SimpleNamespace(setNullSpace=lambda petsc_ns: calls.setdefault("attach", petsc_ns))

    ns.attach_to_matrix(matrix)
    ns.remove_from_vector("rhs")

    assert ns.test_matrix("A") is True
    assert calls["create"] == (True, "WORLD")
    assert calls["attach"] is fake_ns
    assert calls["remove"] == "rhs"
    assert calls["test"] == "A"


def test_neumann_nullspace_handler_checks_rhs_and_fixes_function_gauge():
    ns = NeumannNullspaceHandler(petsc_nullspace=object())

    assert ns.is_rhs_compatible(types.SimpleNamespace(array=np.array([1.0, -1.0]))) is True
    with pytest.raises(ValueError, match="incompatible"):
        ns.check_rhs_compatible(types.SimpleNamespace(array=np.array([1.0, 2.0])))

    scatter_calls = []
    uh = types.SimpleNamespace(
        x=types.SimpleNamespace(
            array=np.array([1.0, 2.0, 3.0]),
            scatter_forward=lambda: scatter_calls.append(True),
        )
    )

    ns.fix_function_gauge(uh)

    assert np.allclose(uh.x.array, [-1.0, 0.0, 1.0])
    assert scatter_calls == [True]


def fake_solver_imports():
    class FakePC:
        class Type:
            NONE = "none"

    class FakeKSP:
        def __init__(self):
            self.pc = types.SimpleNamespace(setType=lambda pc_type: setattr(self, "pc_type", pc_type))

        def create(self, comm):
            self.comm = comm
            return self

        def setOperators(self, A):
            self.A = A

        def setType(self, ksp_type):
            self.ksp_type = ksp_type

        def getPC(self):
            return self.pc

        def setFromOptions(self):
            self.from_options = True

    return {
        "MPI": types.SimpleNamespace(COMM_WORLD="WORLD"),
        "PETSc": types.SimpleNamespace(KSP=FakeKSP, PC=FakePC),
        "fem": types.SimpleNamespace(functionspace=lambda domain, spec: ("V", domain, spec)),
        "fem_petsc": object(),
        "ufl": object(),
    }


def test_neumann_poisson_solver_initializes_from_torso_geometry_with_fakes(monkeypatch):
    mesh = single_tetra_mesh()
    geometry = TorsoGeometry(
        "geom",
        mesh,
        ElectrodeSet(np.zeros((1, 3))),
        SourceRegion.from_cell_ids(mesh, np.array([0], dtype=np.int64)),
    )

    monkeypatch.setattr(neumann_poisson, "require_fenicsx", fake_solver_imports)
    monkeypatch.setattr(
        neumann_poisson, "create_dolfinx_mesh", lambda mesh_data, comm=None: ("domain", mesh_data, comm)
    )
    def fake_assemble(self):
        self.A = "matrix"
        self.K = "matrix"

    monkeypatch.setattr(NeumannPoissonSolver, "_assemble_stiffness", fake_assemble)
    monkeypatch.setattr(NeumannPoissonSolver, "_setup_nullspace", lambda self, test_nullspace: None)

    solver = NeumannPoissonSolver(geometry, degree=1, sigma=2.5, pc_type="none", test_nullspace=False)

    assert solver.mesh_data is mesh
    assert solver.comm == "WORLD"
    assert solver.degree == 1
    assert solver.sigma == 2.5
    assert solver.domain == ("domain", mesh, "WORLD")
    assert solver.V == ("V", solver.domain, ("Lagrange", 1))
    assert solver.K == "matrix"
    assert solver.ksp.ksp_type == "cg"
    assert solver.ksp.pc_type == "none"
    assert solver.diagnostics == SolverDiagnostics("cg", "none", None, None, None)


def test_neumann_poisson_solver_enforces_p1_mvp(monkeypatch):
    monkeypatch.setattr(neumann_poisson, "require_fenicsx", fake_solver_imports)

    with pytest.raises(ValueError, match="P1 Lagrange"):
        NeumannPoissonSolver(single_tetra_mesh(), degree=2)


def test_stiffness_operator_assembles_scalar_constant_conductivity_matrix():
    calls = {}

    class FakeExpr:
        def __init__(self, text):
            self.text = text

        def __mul__(self, other):
            return FakeExpr(f"({self.text}*{other.text if hasattr(other, 'text') else other})")

        def __rmul__(self, other):
            return FakeExpr(f"({other}*{self.text})")

    class FakeUfl:
        @staticmethod
        def TrialFunction(V):
            calls["trial"] = V
            return FakeExpr("u")

        @staticmethod
        def TestFunction(V):
            calls["test"] = V
            return FakeExpr("v")

        @staticmethod
        def dx(domain):
            calls["dx"] = domain
            return FakeExpr("dx")

        @staticmethod
        def grad(expr):
            return FakeExpr(f"grad({expr.text})")

        @staticmethod
        def inner(a, b):
            return FakeExpr(f"inner({a.text},{b.text})")

    class FakeMatrix:
        def assemble(self):
            calls["assemble"] = True

    matrix = FakeMatrix()
    fake_fx = {
        "fem": types.SimpleNamespace(form=lambda expr: calls.setdefault("form", expr)),
        "fem_petsc": types.SimpleNamespace(assemble_matrix=lambda form, bcs: calls.setdefault("assemble_matrix", (form, bcs)) and matrix),
        "ufl": FakeUfl,
    }

    operator = StiffnessOperator(V="V", domain="domain", sigma=2.5, fx=fake_fx)

    assert operator.K is matrix
    assert operator.a_form.text == "((2.5*inner(grad(u),grad(v)))*dx)"
    assert calls["trial"] == "V"
    assert calls["test"] == "V"
    assert calls["dx"] == "domain"
    assert calls["assemble_matrix"] == (operator.a_form, [])
    assert calls["assemble"] is True


def test_linear_solver_reuses_stiffness_matrix_and_records_diagnostics():
    calls = {}

    class FakePC:
        def setType(self, pc_type):
            calls["pc_type"] = pc_type

    class FakeKSP:
        def __init__(self):
            self.pc = FakePC()

        def create(self, comm):
            calls["create"] = comm
            return self

        def setOperators(self, K):
            calls["operators"] = K

        def setType(self, ksp_type):
            calls["ksp_type"] = ksp_type

        def getPC(self):
            return self.pc

        def setFromOptions(self):
            calls["from_options"] = True

        def solve(self, b, x):
            calls["solve"] = (b, x)

        def getConvergedReason(self):
            return 3

        def getResidualNorm(self):
            return 0.125

    fake_fx = {
        "PETSc": types.SimpleNamespace(KSP=FakeKSP, PC=types.SimpleNamespace(Type=types.SimpleNamespace(NONE="none"))),
    }

    solver = LinearSolver(K="K", comm="WORLD", fx=fake_fx, ksp_type="gmres", pc_type="none")
    diagnostics = solver.solve("b", "x")

    assert calls["create"] == "WORLD"
    assert calls["operators"] == "K"
    assert calls["ksp_type"] == "gmres"
    assert calls["pc_type"] == "none"
    assert calls["from_options"] is True
    assert calls["solve"] == ("b", "x")
    assert diagnostics == SolverDiagnostics("gmres", "none", 3, 0.125, None)


def test_neumann_poisson_solver_rejects_unsupported_and_empty_meshes(monkeypatch):
    monkeypatch.setattr(neumann_poisson, "require_fenicsx", fake_solver_imports)
    line = MeshData(points=np.zeros((2, 2)), cells=np.array([[0, 1]]), cell_type="line")
    empty = MeshData(points=np.zeros((3, 2)), cells=np.empty((0, 3), dtype=np.int64), cell_type="triangle")

    with pytest.raises(ValueError, match="triangle or tetra"):
        NeumannPoissonSolver(line)
    with pytest.raises(ValueError, match="at least one cell"):
        NeumannPoissonSolver(empty)


def test_rhs_from_local_array_accepts_matching_shape_and_rejects_mismatch():
    solver = object.__new__(NeumannPoissonSolver)
    scatter_calls = []

    class FakeX:
        def __init__(self):
            self.array = np.zeros(3)

        def scatter_forward(self):
            scatter_calls.append(self.array.copy())

    fake_function = types.SimpleNamespace(x=FakeX())
    solver.zero_function = lambda: fake_function

    result = solver.rhs_from_local_array(np.array([1.0, 2.0, 3.0]))

    assert result is fake_function
    assert np.allclose(fake_function.x.array, [1.0, 2.0, 3.0])
    assert len(scatter_calls) == 1
    with pytest.raises(ValueError, match="local shape"):
        solver.rhs_from_local_array(np.array([1.0, 2.0]))


def test_solve_removes_nullspace_solves_and_updates_diagnostics():
    solver = object.__new__(NeumannPoissonSolver)
    solver.ksp_type = "cg"
    solver.pc_type = "jacobi"
    solver.diagnostics = SolverDiagnostics("cg", "jacobi", None, None, True)

    calls = {}

    class Vec:
        def __init__(self, name):
            self.name = name

        def copy(self):
            calls["copy"] = self.name
            return Vec(self.name + "_copy")

        def set(self, value):
            calls["set"] = value

        def destroy(self):
            calls["destroy"] = self.name

    class FakeNullspace:
        def remove_from_vector(self, b):
            calls["remove"] = b.name

        def check_rhs_compatible(self, b, *, tol):
            calls["check_rhs"] = (b.name, tol)

        def fix_function_gauge(self, uh):
            calls["fix_gauge"] = uh.x.petsc_vec.name

    class FakeLinearSolver:
        def solve(self, b, x):
            calls["solve"] = (b.name, x.name)
            return SolverDiagnostics("cg", "jacobi", 2, 1.25e-8, None)

    solution_vec = Vec("solution")
    solver.nullspace = FakeNullspace()
    solver.linear_solver = FakeLinearSolver()
    solver.ksp = object()
    solver.rhs_compatibility_tol = 1e-10
    solver.zero_function = lambda: types.SimpleNamespace(
        x=types.SimpleNamespace(petsc_vec=solution_vec, scatter_forward=lambda: calls.setdefault("scatter", True))
    )

    result = solver.solve(types.SimpleNamespace(x=types.SimpleNamespace(petsc_vec=Vec("rhs"))))

    assert result.x.petsc_vec is solution_vec
    assert calls == {
        "copy": "rhs",
        "remove": "rhs_copy",
        "check_rhs": ("rhs_copy", 1e-10),
        "set": 0.0,
        "solve": ("rhs_copy", "solution"),
        "scatter": True,
        "fix_gauge": "solution",
        "destroy": "rhs_copy",
    }
    assert solver.diagnostics == SolverDiagnostics("cg", "jacobi", 2, 1.25e-8, True)


def test_solve_raises_on_nonconvergence_when_requested():
    solver = object.__new__(NeumannPoissonSolver)
    solver.ksp_type = "gmres"
    solver.pc_type = "none"
    solver.diagnostics = SolverDiagnostics("gmres", "none", None, None, None)
    solver.rhs_compatibility_tol = 1e-10
    solver.nullspace = types.SimpleNamespace(
        remove_from_vector=lambda b: None,
        check_rhs_compatible=lambda b, tol: None,
        fix_function_gauge=lambda uh: None,
    )

    class Vec:
        def copy(self):
            return self

        def set(self, value):
            pass

        def destroy(self):
            pass

    solver.linear_solver = types.SimpleNamespace(solve=lambda b, x: SolverDiagnostics("gmres", "none", -3, 4.0, None))
    solver.ksp = object()
    solver.zero_function = lambda: types.SimpleNamespace(
        x=types.SimpleNamespace(petsc_vec=Vec(), scatter_forward=lambda: None)
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        solver.solve(types.SimpleNamespace(x=types.SimpleNamespace(petsc_vec=Vec())))
