import numpy as np
import pytest

from geometry import ElectrodeSet, MeshData
from measurements import (
    CentralSurfaceProjector,
    apply_reference,
    boundary_triangle_mesh_from_tetra_mesh,
    build_measurement_operator,
    build_point_interpolation_matrix,
    central_project_electrodes_to_surface,
    evaluate_at_points,
    locate_electrodes_in_mesh,
    locate_points_in_tetra_mesh,
    measure_nodal_values,
    measure_raw_nodal_values,
    TetraVolumeLocator,
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


def test_locate_points_in_tetra_mesh_returns_cell_ids_and_barycentric_coordinates():
    mesh = single_tetra_mesh()
    points = np.array([[0.25, 0.25, 0.25]])

    cell_ids, barycentric = locate_points_in_tetra_mesh(mesh, points)

    assert np.array_equal(cell_ids, [0])
    assert barycentric.shape == (1, 4)
    assert np.allclose(barycentric[0], [0.25, 0.25, 0.25, 0.25])


def test_locate_points_in_tetra_mesh_rejects_outside_point():
    mesh = single_tetra_mesh()

    with pytest.raises(ValueError, match="point 0"):
        locate_points_in_tetra_mesh(mesh, np.array([[2.0, 0.0, 0.0]]))


def test_central_project_electrodes_to_inferred_surface():
    mesh = single_tetra_mesh()
    electrodes = ElectrodeSet(positions=np.array([[2.0, 0.0, 0.0], [0.25, 0.25, 0.25]]))

    projected, report = central_project_electrodes_to_surface(mesh, electrodes)

    assert report.num_projected == 1
    assert np.array_equal(report.projected_indices, [0])
    assert np.allclose(projected.positions[0], [0.6, 0.2, 0.2])
    assert np.allclose(projected.positions[1], electrodes.positions[1])
    assert projected.metadata["projection"]["num_projected"] == 1


def test_tetra_volume_locator_contains_points_and_marks_outside():
    mesh = single_tetra_mesh()
    locator = TetraVolumeLocator(mesh)
    points = np.array(
        [
            [0.25, 0.25, 0.25],
            [2.0, 0.0, 0.0],
        ]
    )

    cell_ids = locator.locate_points(points)

    assert np.array_equal(cell_ids, [0, -1])
    assert np.array_equal(locator.contains_points(points), [True, False])


def test_central_surface_projector_projects_point():
    mesh = single_tetra_mesh()
    surface = boundary_triangle_mesh_from_tetra_mesh(mesh)
    projector = CentralSurfaceProjector(surface, center=mesh.points.mean(axis=0))

    projected, surface_cell_id = projector.project_point([2.0, 0.0, 0.0])

    assert surface_cell_id >= 0
    assert np.allclose(projected, [0.6, 0.2, 0.2])


def test_central_projection_reuses_volume_locator_for_batch_inside_check():
    mesh = single_tetra_mesh()
    electrodes = ElectrodeSet(
        positions=np.array(
            [
                [2.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.0, 2.0, 0.0],
            ]
        )
    )
    locator = TetraVolumeLocator(mesh)
    calls = {"contains_points": 0}
    original_contains_points = locator.contains_points

    def counted_contains_points(points):
        calls["contains_points"] += 1
        return original_contains_points(points)

    locator.contains_points = counted_contains_points

    projected, report = central_project_electrodes_to_surface(
        mesh,
        electrodes,
        volume_locator=locator,
    )

    assert calls["contains_points"] == 1
    assert report.num_projected == 2
    assert np.array_equal(report.projected_indices, [0, 2])
    assert np.allclose(projected.positions[1], electrodes.positions[1])


def test_locate_electrodes_can_project_outside_positions():
    mesh = single_tetra_mesh()
    electrodes = ElectrodeSet(positions=np.array([[2.0, 0.0, 0.0]]), labels=["outside"])

    cell_ids, barycentric, projected, report = locate_electrodes_in_mesh(
        mesh,
        electrodes,
        project_outside=True,
        return_projection=True,
    )

    assert np.array_equal(cell_ids, [0])
    assert np.allclose(projected.positions[0], [0.6, 0.2, 0.2])
    assert np.allclose(barycentric[0], [0.0, 0.6, 0.2, 0.2], atol=1e-12)
    assert report is not None
    assert report.num_projected == 1


def test_build_point_interpolation_matrix_dense():
    mesh = single_tetra_mesh()
    points = np.array([[0.25, 0.25, 0.25]])

    matrix = build_point_interpolation_matrix(mesh, points, sparse=False)

    assert matrix.shape == (1, 4)
    assert np.allclose(matrix[0], [0.25, 0.25, 0.25, 0.25])


def test_evaluate_at_points_uses_p1_interpolation():
    mesh = single_tetra_mesh()
    nodal_values = np.array([0.0, 1.0, 2.0, 3.0])

    values = evaluate_at_points(mesh, nodal_values, np.array([[0.25, 0.25, 0.25]]))

    assert values.shape == (1,)
    assert values[0] == pytest.approx(1.5)


def test_multiple_electrode_raw_values():
    mesh = single_tetra_mesh()
    electrodes = ElectrodeSet(
        positions=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
            ]
        ),
        labels=["E1", "E2"],
    )
    nodal_values = np.array([0.0, 1.0, 2.0, 3.0])

    values = measure_raw_nodal_values(mesh, electrodes, nodal_values)

    assert np.allclose(values, [1.0, 1.5])


def test_average_reference_values_sum_to_zero():
    values = np.array([1.0, 2.0, 3.0])

    referenced = apply_reference(values, reference="average")

    assert np.allclose(referenced, [-1.0, 0.0, 1.0])
    assert referenced.sum() == pytest.approx(0.0)


def test_single_reference_sets_reference_electrode_to_zero():
    values = np.array([1.0, 2.0, 3.0])

    referenced = apply_reference(values, reference="single", reference_index=1)

    assert np.allclose(referenced, [-1.0, 0.0, 1.0])
    assert referenced[1] == pytest.approx(0.0)


def test_measurement_operator_evaluates_raw_and_referenced_values():
    mesh = single_tetra_mesh()
    electrodes = ElectrodeSet(
        positions=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
            ]
        ),
        labels=["E1", "E2"],
    )
    nodal_values = np.array([0.0, 1.0, 2.0, 3.0])

    op = build_measurement_operator(mesh, electrodes, reference="average", sparse=False)

    assert np.allclose(op.evaluate_raw(nodal_values), [1.0, 1.5])
    assert np.allclose(op.evaluate(nodal_values), [-0.25, 0.25])
    assert op.matrix().shape == (2, 4)
    assert op.num_electrodes == 2
    assert op.num_nodes == 4
    assert op.labels == ["E1", "E2"]


def test_measurement_operator_projects_outside_electrodes_by_default():
    mesh = single_tetra_mesh()
    electrodes = ElectrodeSet(positions=np.array([[2.0, 0.0, 0.0]]), labels=["outside"])
    nodal_values = np.array([0.0, 1.0, 2.0, 3.0])

    op = build_measurement_operator(mesh, electrodes, reference="none", sparse=False)

    assert np.allclose(op.electrode_barycentric[0], [0.0, 0.6, 0.2, 0.2], atol=1e-12)
    assert op.evaluate_raw(nodal_values) == pytest.approx([1.6])
    assert op.metadata["electrode_projection"]["num_projected"] == 1
    assert op.metadata["electrode_projection"]["projected_indices"] == [0]


def test_average_referenced_constant_function_is_zero():
    mesh = single_tetra_mesh()
    electrodes = ElectrodeSet(
        positions=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.0, 1.0, 0.0],
            ]
        )
    )
    nodal_values = np.array([5.0, 5.0, 5.0, 5.0])

    measured = measure_nodal_values(mesh, electrodes, nodal_values, reference="average")

    assert np.allclose(measured, [0.0, 0.0, 0.0])


def test_average_reference_is_invariant_to_constant_nodal_shift():
    mesh = single_tetra_mesh()
    electrodes = ElectrodeSet(
        positions=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.0, 1.0, 0.0],
            ]
        )
    )
    op = build_measurement_operator(mesh, electrodes, reference="average", sparse=False)
    nodal_values = np.array([0.0, 1.0, 2.0, 3.0])

    measured = op.evaluate(nodal_values)
    shifted = op.evaluate(nodal_values + 17.5)

    assert np.allclose(shifted, measured)


def test_sparse_and_dense_measurement_matrices_are_consistent():
    pytest.importorskip("scipy")
    mesh = single_tetra_mesh()
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ]
    )
    electrodes = ElectrodeSet(positions=points)

    p_sparse = build_point_interpolation_matrix(mesh, points, sparse=True)
    p_dense = build_point_interpolation_matrix(mesh, points, sparse=False)
    op_sparse = build_measurement_operator(mesh, electrodes, reference="average", sparse=True)
    op_dense = build_measurement_operator(mesh, electrodes, reference="average", sparse=False)

    assert np.allclose(p_sparse.toarray(), p_dense)
    assert np.allclose(op_sparse.matrix().toarray(), op_dense.matrix())
