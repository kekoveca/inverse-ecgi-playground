import numpy as np
import pytest

from geometry import MeshData
from sources import (
    PointDipole,
    assemble_point_dipole_rhs_numpy,
    barycentric_boundary_flags,
    barycentric_coordinates_tetra,
    check_rhs_compatibility,
    gradients_p1_tetra,
    locate_point_in_mesh,
    locate_points_in_mesh,
    point_in_tetra,
    rhs_compatibility_error,
    tetra_signed_volume,
    tetra_volume,
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
    return MeshData(
        points=standard_vertices(),
        cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
        cell_type="tetra",
    )


def two_tetra_mesh():
    points = np.vstack(
        [
            standard_vertices(),
            standard_vertices() + np.array([2.0, 0.0, 0.0]),
        ]
    )
    return MeshData(
        points=points,
        cells=np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64),
        cell_type="tetra",
    )


def test_barycentric_coordinates_tetra_standard_tetra():
    lambdas = barycentric_coordinates_tetra(np.array([0.25, 0.25, 0.25]), standard_vertices())

    assert np.allclose(lambdas, [0.25, 0.25, 0.25, 0.25])
    assert np.isclose(lambdas.sum(), 1.0)


def test_barycentric_boundary_flags_classify_interior_face_edge_and_vertex():
    interior = barycentric_boundary_flags(np.array([0.25, 0.25, 0.25, 0.25]))
    face = barycentric_boundary_flags(np.array([0.0, 0.25, 0.25, 0.5]))
    edge = barycentric_boundary_flags(np.array([0.0, 0.0, 0.25, 0.75]))
    vertex = barycentric_boundary_flags(np.array([1.0, 0.0, 0.0, 0.0]))

    assert interior["is_on_boundary"] is False
    assert interior["boundary_kind"] == "interior"
    assert face["is_on_boundary"] is True
    assert face["boundary_kind"] == "face"
    assert np.array_equal(face["near_zero_indices"], [0])
    assert edge["boundary_kind"] == "edge"
    assert np.array_equal(edge["near_zero_indices"], [0, 1])
    assert vertex["boundary_kind"] == "vertex"
    assert np.array_equal(vertex["near_one_indices"], [0])


def test_point_in_tetra_detects_inside_and_outside_points():
    vertices = standard_vertices()

    assert point_in_tetra([0.25, 0.25, 0.25], vertices)
    assert not point_in_tetra([2.0, 0.0, 0.0], vertices)


def test_gradients_p1_tetra_standard_tetra():
    grads = gradients_p1_tetra(standard_vertices())
    expected = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert np.allclose(grads, expected)
    assert np.allclose(grads.sum(axis=0), [0.0, 0.0, 0.0])


def test_tetra_volume_helpers():
    vertices = standard_vertices()

    assert tetra_signed_volume(vertices) == pytest.approx(1.0 / 6.0)
    assert tetra_volume(vertices) == pytest.approx(1.0 / 6.0)


def test_locate_point_in_mesh_finds_cell_and_rejects_outside_point():
    mesh = single_tetra_mesh()

    assert locate_point_in_mesh(mesh, np.array([0.25, 0.25, 0.25])) == 0
    with pytest.raises(ValueError, match="not inside"):
        locate_point_in_mesh(mesh, np.array([2.0, 0.0, 0.0]))


def test_locate_points_in_mesh_uses_batched_kdtree_search():
    mesh = two_tetra_mesh()
    points = np.array(
        [
            [0.25, 0.25, 0.25],
            [2.25, 0.25, 0.25],
        ]
    )

    assert np.array_equal(locate_points_in_mesh(mesh, points), [0, 1])
    assert np.array_equal(locate_points_in_mesh(mesh, points[1:], candidate_cell_ids=[1]), [1])
    with pytest.raises(ValueError, match="not inside"):
        locate_points_in_mesh(mesh, points[:1], candidate_cell_ids=[1])


def test_assemble_point_dipole_rhs_numpy_standard_tetra():
    mesh = single_tetra_mesh()
    source = PointDipole(position=np.array([0.25, 0.25, 0.25]), moment=np.array([1.0, 2.0, 3.0]))

    rhs = assemble_point_dipole_rhs_numpy(mesh, source)

    assert rhs.shape == (4,)
    assert np.allclose(rhs, [-6.0, 1.0, 2.0, 3.0])
    assert np.isclose(rhs.sum(), 0.0)


def test_rhs_compatibility_helpers_accept_dipole_rhs():
    mesh = single_tetra_mesh()
    source = PointDipole(position=[0.25, 0.25, 0.25], moment=[1.0, 2.0, 3.0])
    rhs = assemble_point_dipole_rhs_numpy(mesh, source)

    assert check_rhs_compatibility(rhs) is True
    assert rhs_compatibility_error(rhs) == pytest.approx(0.0)


def test_point_dipole_with_cell_id_returns_new_source():
    source = PointDipole(position=[0.25, 0.25, 0.25], moment=[1.0, 0.0, 0.0])
    source2 = source.with_cell_id(0)

    assert source.cell_id is None
    assert source2.cell_id == 0
    assert np.allclose(source.normalized_moment(), [1.0, 0.0, 0.0])


def test_degenerate_tetra_raises_clear_error():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="degenerate tetrahedron"):
        gradients_p1_tetra(vertices)
    with pytest.raises(ValueError, match="degenerate tetrahedron"):
        barycentric_coordinates_tetra([0.0, 0.0, 0.0], vertices)


def test_point_dipole_rejects_malformed_values_and_zero_normalization():
    with pytest.raises(ValueError, match="position must have shape"):
        PointDipole(position=[0.0, 0.0], moment=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        PointDipole(position=[0.0, 0.0, 0.0], moment=[np.nan, 0.0, 0.0])
    with pytest.raises(ValueError, match="zero dipole moment"):
        PointDipole(position=[0.0, 0.0, 0.0], moment=[0.0, 0.0, 0.0]).normalized_moment()
