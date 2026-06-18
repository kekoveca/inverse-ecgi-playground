import os

import numpy as np
import pytest

from forward import ForwardResult
from sources import PointDipole


os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("OMPI_MCA_orte_tmpdir_base", "/tmp")

DOLFINX_TESTS_ENABLED = os.environ.get("RUN_DOLFINX_TESTS") == "1"
requires_dolfinx = pytest.mark.skipif(
    not DOLFINX_TESTS_ENABLED,
    reason="set RUN_DOLFINX_TESTS=1 to run real DOLFINx forward tests",
)


def standard_vertices():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def single_tetra_mesh():
    from geometry import MeshData

    return MeshData(
        points=standard_vertices(),
        cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
        cell_type="tetra",
    )


def two_electrodes():
    from geometry import ElectrodeSet

    return ElectrodeSet(
        positions=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        labels=["E1", "E2"],
    )


def test_forward_result_properties():
    result = ForwardResult(
        source=PointDipole(position=[0.25, 0.25, 0.25], moment=[1.0, 0.0, 0.0]),
        potential=None,
        nodal_values=np.array([1.0, 2.0, 3.0]),
        raw_measurements=np.array([1.0, 2.0]),
        measurements=np.array([-0.5, 0.5]),
        reference="average",
    )

    assert result.num_nodes == 3
    assert result.num_electrodes == 2
    assert result.measurement_norm == pytest.approx(np.linalg.norm(result.measurements))
    assert result.raw_measurement_norm == pytest.approx(np.linalg.norm(result.raw_measurements))


def test_forward_result_to_dict_omits_large_arrays():
    result = ForwardResult(
        source=PointDipole(position=[0.25, 0.25, 0.25], moment=[1.0, 0.0, 0.0]),
        potential=None,
        nodal_values=np.array([1.0, 2.0, 3.0]),
        raw_measurements=np.array([1.0, 2.0]),
        measurements=np.array([-0.5, 0.5]),
        reference="average",
        metadata={"case": "unit"},
    )

    summary = result.to_dict()

    assert summary["num_nodes"] == 3
    assert summary["num_electrodes"] == 2
    assert summary["measurement_norm"] == pytest.approx(result.measurement_norm)
    assert summary["raw_measurement_norm"] == pytest.approx(result.raw_measurement_norm)
    assert summary["reference"] == "average"
    assert summary["metadata"] == {"case": "unit"}
    assert "nodal_values" not in summary
    assert "raw_measurements" not in summary
    assert "measurements" not in summary


def _forward_result_for_one_tetra():
    pytest.importorskip("dolfinx")
    pytest.importorskip("mpi4py")
    pytest.importorskip("petsc4py")

    from fem import NeumannPoissonSolver
    from forward import ForwardSolver

    problem = NeumannPoissonSolver(single_tetra_mesh(), pc_type="none", test_nullspace=True)
    forward = ForwardSolver(
        poisson_solver=problem,
        electrodes=two_electrodes(),
        reference="average",
        measurement_sparse=True,
    )
    source = PointDipole(position=[0.25, 0.25, 0.25], moment=[1.0, 0.0, 0.0])
    return problem, forward.solve(source)


@requires_dolfinx
def test_forward_solve_on_one_tetra():
    problem, result = _forward_result_for_one_tetra()
    try:
        assert result.nodal_values.shape[0] == 4
        assert result.raw_measurements.shape == (2,)
        assert result.measurements.shape == (2,)
        assert result.measurements.sum() == pytest.approx(0.0, abs=1e-12)
    finally:
        problem.destroy()


@requires_dolfinx
def test_extract_nodal_values_from_dolfinx_function():
    problem, result = _forward_result_for_one_tetra()
    try:
        from forward import extract_nodal_values

        values = extract_nodal_values(result.potential)

        assert isinstance(values, np.ndarray)
        assert values.shape[0] == 4
        assert np.allclose(values, result.nodal_values)
        values[:] = 100.0
        assert not np.allclose(values, result.potential.x.array)
    finally:
        problem.destroy()


@requires_dolfinx
def test_export_xdmf_creates_file(tmp_path):
    problem, result = _forward_result_for_one_tetra()
    try:
        from forward import export_forward_result_to_xdmf

        path = export_forward_result_to_xdmf(result, tmp_path / "potential.xdmf")

        assert path.exists()
        assert path.suffix == ".xdmf"
    finally:
        problem.destroy()


@requires_dolfinx
def test_export_vtx_creates_output(tmp_path):
    pytest.importorskip("dolfinx")
    from dolfinx import io

    if not hasattr(io, "VTXWriter"):
        pytest.skip("dolfinx.io.VTXWriter is unavailable")

    problem, result = _forward_result_for_one_tetra()
    try:
        from forward import export_forward_result_to_vtx

        try:
            path = export_forward_result_to_vtx(result, tmp_path / "potential.bp")
        except RuntimeError as exc:
            pytest.skip(f"VTXWriter/ADIOS2 runtime is unavailable: {exc}")
        except ImportError as exc:
            pytest.skip(str(exc))

        assert path.exists()
        assert path.suffix == ".bp"
    finally:
        problem.destroy()


def test_export_rejects_non_dolfinx_function(tmp_path):
    from forward import export_potential_to_xdmf

    with pytest.raises(TypeError, match="dolfinx.fem.Function"):
        export_potential_to_xdmf(object(), tmp_path / "bad.xdmf")
