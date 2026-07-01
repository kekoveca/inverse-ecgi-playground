import numpy as np
import pytest

from geometry import tetra_volumes
from sources import locate_point_in_mesh
from verification import (
    build_convergence_report,
    create_unit_cube_meshdata,
    estimate_rates,
    format_convergence_report,
    homogeneous_free_space_dipole_potential,
    rhs_neumann_cosine,
    u_exact_neumann_cosine,
)


def test_estimate_rates_recovers_quadratic_order():
    h = np.array([0.25, 0.125, 0.0625])
    errors = h**2

    rates = estimate_rates(h, errors)
    report = build_convergence_report(h, errors)

    assert np.allclose(rates, [2.0, 2.0])
    assert report.errors_decrease is True
    assert report.min_rate == pytest.approx(2.0)
    assert "level | h | error | rate" in format_convergence_report(report)


def test_convergence_report_validates_input():
    with pytest.raises(ValueError, match="strictly decreasing"):
        estimate_rates([0.5, 0.5], [0.25, 0.125])
    with pytest.raises(ValueError, match="positive"):
        estimate_rates([0.5, 0.25], [0.25, 0.0])


def test_neumann_cosine_manufactured_functions():
    points = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])

    exact = u_exact_neumann_cosine(points)
    rhs = rhs_neumann_cosine(points)

    assert np.allclose(exact, [1.0, 0.0])
    assert np.allclose(rhs, 12.0 * np.pi**2 * exact)


def test_homogeneous_dipole_reference_has_expected_axis_sign():
    values = homogeneous_free_space_dipole_potential(
        points=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        position=[0.0, 0.0, 0.0],
        moment=[0.0, 0.0, 1.0],
    )

    assert values[0] == pytest.approx(1.0 / (4.0 * np.pi))
    assert values[1] == pytest.approx(-1.0 / (4.0 * np.pi))


def test_create_unit_cube_meshdata_is_valid_and_contains_center():
    n = 2
    mesh = create_unit_cube_meshdata(n)
    volumes = tetra_volumes(mesh)

    assert mesh.cell_type == "tetra"
    assert mesh.num_points == (n + 1) ** 3
    assert mesh.num_cells == 6 * n**3
    assert np.all(volumes > 0.0)
    assert np.sum(volumes) == pytest.approx(1.0)
    assert locate_point_in_mesh(mesh, [0.5, 0.5, 0.5]) >= 0
