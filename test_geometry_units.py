import sys
import types

import numpy as np
import pytest

import geometry
from geometry.electrodes import ElectrodeSet, electrode_placement_report
from geometry.mesh_model import MeshData, load_npz_mesh, quality_report, save_npz_mesh, tetra_volumes
from geometry.source_region import SourceRegion
from geometry.tagged_mesh import Mesh, TaggedMesh, _field_data_to_tuples, read_gmsh_meshio
from geometry.torso_geometry import TorsoGeometry
from geometry.transforms import (
    AffineTransform,
    transform_electrodes,
    transform_mesh,
    transform_source_region,
    transform_torso_geometry,
)
from geometry.validation import validate_torso_geometry
from geometry.visualization import plot_mesh, plot_source_region, plot_torso_geometry


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


def simple_geometry():
    mesh = single_tetra_mesh()
    electrodes = ElectrodeSet(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        labels=["E1", "E2"],
        metadata={"unit": "m"},
    )
    source_region = SourceRegion.from_cell_ids(mesh, np.array([0], dtype=np.int64), metadata={"kind": "center"})
    surface_mesh = MeshData(
        points=mesh.points,
        cells=np.array([[0, 1, 2]], dtype=np.int64),
        cell_type="triangle",
        name="surface",
    )
    return TorsoGeometry("geom", mesh, electrodes, source_region, surface_mesh=surface_mesh, metadata={"case": "demo"})


def test_public_geometry_exports_are_available():
    for name in geometry.__all__:
        assert hasattr(geometry, name), name
    assert geometry.Mesh is geometry.TaggedMesh


def test_mesh_data_properties_centers_metadata_and_npz_roundtrip(tmp_path):
    mesh = single_tetra_mesh().with_metadata(patient="demo")

    assert mesh.geometric_dim == 3
    assert mesh.num_points == 4
    assert mesh.num_cells == 1
    assert np.allclose(mesh.bounding_box()[0], [0.0, 0.0, 0.0])
    assert np.allclose(mesh.bounding_box()[1], [1.0, 1.0, 1.0])
    assert np.allclose(mesh.cell_centers(), [[0.25, 0.25, 0.25]])
    assert mesh.metadata == {"patient": "demo"}

    path = tmp_path / "mesh.npz"
    save_npz_mesh(mesh, path)
    loaded = load_npz_mesh(path)

    assert loaded.name == "single_tet"
    assert loaded.cell_type == "tetra"
    assert loaded.metadata == {"patient": "demo"}
    assert np.allclose(loaded.points, mesh.points)
    assert np.array_equal(loaded.cells, mesh.cells)


def test_mesh_quality_reports_tetra_volumes_and_degenerate_cells():
    mesh = single_tetra_mesh()
    assert np.allclose(tetra_volumes(mesh), [1.0 / 6.0])

    report = quality_report(mesh)
    assert report.min_cell_volume == pytest.approx(1.0 / 6.0)
    assert report.max_cell_volume == pytest.approx(1.0 / 6.0)
    assert report.mean_cell_volume == pytest.approx(1.0 / 6.0)
    assert report.num_degenerate_cells == 0

    degenerate = MeshData(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ),
        cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
        cell_type="tetra",
    )
    assert quality_report(degenerate).num_degenerate_cells == 1


def test_mesh_data_rejects_bad_shapes_and_indices():
    with pytest.raises(ValueError, match="points must have shape"):
        MeshData(points=np.array([0.0, 1.0]), cells=np.array([[0, 1]]), cell_type="line")
    with pytest.raises(ValueError, match="outside points array"):
        MeshData(points=np.zeros((2, 2)), cells=np.array([[0, 2]]), cell_type="line")
    with pytest.raises(ValueError, match="requires 3 nodes"):
        MeshData(points=np.zeros((3, 2)), cells=np.array([[0, 1]]), cell_type="triangle")
    with pytest.raises(ValueError, match="requires cell_type='tetra'"):
        tetra_volumes(MeshData(points=np.zeros((2, 2)), cells=np.array([[0, 1]]), cell_type="line"))


def test_electrode_set_labels_centering_and_nearest_mesh_nodes():
    electrodes = ElectrodeSet(np.array([[0.0, 0.0, 0.0], [0.9, 0.1, 0.0]]))
    mesh = single_tetra_mesh()

    assert electrodes.labels == ["E000", "E001"]
    assert np.array_equal(electrodes.nearest_mesh_nodes(mesh), [0, 1])
    assert np.allclose(electrodes.distance_to_mesh_nodes(mesh), [0.0, np.sqrt(0.02)])

    centered = electrodes.centered()
    assert np.allclose(centered.positions.mean(axis=0), [0.0, 0.0, 0.0])
    assert centered.labels == electrodes.labels

    report = electrode_placement_report(electrodes, mesh)
    assert np.array_equal(report.nearest_node_ids, [0, 1])
    assert report.mean_distance_to_nearest_node == pytest.approx(np.sqrt(0.02) / 2.0)
    assert report.max_distance_to_nearest_node == pytest.approx(np.sqrt(0.02))


def test_electrode_set_rejects_malformed_inputs_and_dimension_mismatch():
    with pytest.raises(ValueError, match="positions must have shape"):
        ElectrodeSet(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="labels length"):
        ElectrodeSet(np.zeros((2, 3)), labels=["E1"])
    with pytest.raises(ValueError, match="same geometric dimension"):
        ElectrodeSet(np.zeros((1, 2))).nearest_mesh_nodes(single_tetra_mesh())


def test_source_region_builders_bbox_modes_and_subset():
    mesh = MeshData(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, 0.0, 1.0],
            ]
        ),
        cells=np.array([[0, 1, 2, 3], [1, 4, 5, 6]], dtype=np.int64),
        cell_type="tetra",
        name="two_tets",
    )

    all_cells = SourceRegion.all_cells(mesh)
    assert np.array_equal(all_cells.candidate_cell_ids, [0, 1])

    center = SourceRegion.from_bounding_box(mesh, [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], mode="center")
    any_vertex = SourceRegion.from_bounding_box(mesh, [0.0, 0.0, 0.0], [0.1, 0.1, 0.1], mode="any_vertex")
    all_vertices = SourceRegion.from_bounding_box(mesh, [0.0, 0.0, 0.0], [2.0, 1.0, 1.0], mode="all_vertices")

    assert np.array_equal(center.candidate_cell_ids, [0])
    assert np.array_equal(any_vertex.candidate_cell_ids, [0])
    assert np.array_equal(all_vertices.candidate_cell_ids, [0, 1])
    assert center.metadata["selection"] == "bounding_box"
    assert center.metadata["source_mesh_name"] == "two_tets"

    subset = all_cells.subset(np.array([1]), name="one")
    assert subset.name == "one"
    assert np.array_equal(subset.candidate_cell_ids, [1])
    assert np.allclose(subset.candidate_points, [[1.75, 0.25, 0.25]])


def test_source_region_rejects_invalid_ids_bounds_and_shapes():
    mesh = single_tetra_mesh()
    with pytest.raises(ValueError, match="cell_ids must be one-dimensional"):
        SourceRegion.from_cell_ids(mesh, np.array([[0]]))
    with pytest.raises(ValueError, match="invalid mesh cell indices"):
        SourceRegion.from_cell_ids(mesh, np.array([1]))
    with pytest.raises(ValueError, match="bounds_min must have shape"):
        SourceRegion.from_bounding_box(mesh, [0.0, 0.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="bounds_min must be <= bounds_max"):
        SourceRegion.from_bounding_box(mesh, [1.0, 0.0, 0.0], [0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="mode must be one of"):
        SourceRegion.from_bounding_box(mesh, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], mode="bad")


def test_tagged_mesh_filters_physical_groups_and_converts_to_mesh_data():
    tagged = TaggedMesh(
        dim=2,
        coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        cells={
            "triangle": np.array([[0, 1, 2], [1, 3, 2]]),
            "line": np.array([[0, 1]]),
        },
        cell_tags={"triangle": np.array([7, 8])},
        field_data={"torso": (2, 7), "boundary": (1, 3)},
        metadata={"origin": "gmsh"},
    )

    assert isinstance(tagged, Mesh)
    assert tagged.num_points == 4
    assert tagged.field_data["torso"] == (2, 7)
    assert tagged.physical_tag("torso") == 7
    assert tagged.physical_dimension("boundary") == 1
    assert np.array_equal(tagged.tags_for("line"), [0])
    assert np.array_equal(tagged.cell_block("triangle", physical_name="torso"), [[0, 1, 2]])

    mesh = tagged.to_mesh_data("triangle", name="domain", physical_name="torso")
    assert mesh.name == "domain"
    assert mesh.metadata["source"] == "TaggedMesh"
    assert mesh.metadata["origin"] == "gmsh"
    assert mesh.metadata["physical_dimension"] == 2
    assert mesh.metadata["physical_tag"] == 7
    assert np.array_equal(mesh.cells, [[0, 1, 2]])


def test_tagged_mesh_rejects_invalid_data_and_unknown_groups():
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        TaggedMesh(4, np.zeros((1, 4)), {}, {}, {})
    with pytest.raises(ValueError, match="must have shape"):
        TaggedMesh(2, np.zeros((2, 2)), {"line": np.array([[0, 1]])}, {"line": np.array([1, 2])}, {})
    with pytest.raises(ValueError, match="at least tag and dimension"):
        _field_data_to_tuples({"bad": [1]})

    tagged = TaggedMesh(2, np.zeros((2, 2)), {"line": np.array([[0, 1]])}, {}, {})
    with pytest.raises(KeyError, match="Physical group"):
        tagged.physical_tag("missing")
    with pytest.raises(KeyError, match="Cell type"):
        tagged.cell_block("triangle")


def test_read_gmsh_meshio_maps_meshio_data(monkeypatch, tmp_path):
    fake_mesh = types.SimpleNamespace(
        points=np.array([[0.0, 0.0, 9.0], [1.0, 0.0, 9.0], [0.0, 1.0, 9.0]]),
        cells_dict={"triangle": np.array([[0, 1, 2]], dtype=np.int64)},
        cell_data_dict={"gmsh:physical": {"triangle": np.array([7], dtype=np.int64)}},
        field_data={"torso": np.array([7, 2])},
    )
    fake_meshio = types.SimpleNamespace(read=lambda path: fake_mesh)
    monkeypatch.setitem(sys.modules, "meshio", fake_meshio)

    tagged = read_gmsh_meshio(tmp_path / "demo.msh", dim=2)

    assert tagged.dim == 2
    assert np.allclose(tagged.coords, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert np.array_equal(tagged.cell_tags["triangle"], [7])
    assert tagged.field_data["torso"] == (2, 7)
    assert tagged.physical_tag("torso") == 7
    assert tagged.physical_dimension("torso") == 2
    assert tagged.metadata["reader"] == "meshio"


def test_meshio_field_data_is_converted_to_internal_dimension_tag_order():
    converted = _field_data_to_tuples(
        {
            "boundary": np.array([2, 2]),
            "domain": np.array([1, 3]),
        }
    )

    assert converted["domain"] == (3, 1)
    assert converted["boundary"] == (2, 2)


def test_tagged_mesh_uses_internal_dimension_tag_order_for_manual_data():
    tagged = Mesh(
        dim=3,
        coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        cells={
            "tetra": np.array([[0, 1, 2, 3]]),
        },
        cell_tags={
            "tetra": np.array([1]),
        },
        field_data={
            "domain": (3, 1),
        },
    )

    assert tagged.field_data["domain"] == (3, 1)
    assert tagged.physical_dimension("domain") == 3
    assert tagged.physical_tag("domain") == 1

    domain = tagged.to_mesh_data(
        cell_type="tetra",
        physical_name="domain",
    )

    assert domain.num_cells == 1
    assert domain.metadata["physical_dimension"] == 3
    assert domain.metadata["physical_tag"] == 1


def test_tagged_mesh_filters_cells_by_physical_tag_not_dimension():
    tagged = TaggedMesh(
        dim=3,
        coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        cells={
            "tetra": np.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
        },
        cell_tags={
            "tetra": np.array([1, 1]),
        },
        field_data={"domain": (3, 1)},
    )

    assert tagged.cell_block("tetra", physical_name="domain").shape[0] == 2
    assert tagged.to_mesh_data("tetra", physical_name="domain").num_cells == 2


def test_torso_geometry_summary_and_quality_report():
    geom = simple_geometry()

    assert geom.quality_report().num_cells == 1
    assert geom.summary() == {
        "geometry_id": "geom",
        "num_volume_points": 4,
        "num_volume_cells": 1,
        "num_electrodes": 2,
        "num_source_candidates": 1,
        "geometric_dim": 3,
        "min_cell_volume": pytest.approx(1.0 / 6.0),
        "num_degenerate_cells": 0,
    }


def test_torso_geometry_rejects_dimension_and_cell_id_mismatches():
    mesh = single_tetra_mesh()
    source_region = SourceRegion.from_cell_ids(mesh, np.array([0]))
    with pytest.raises(ValueError, match="electrodes and volume mesh dimensions"):
        TorsoGeometry("bad", mesh, ElectrodeSet(np.zeros((1, 2))), source_region)
    with pytest.raises(ValueError, match="source region has cell ids outside"):
        TorsoGeometry(
            "bad",
            mesh,
            ElectrodeSet(np.zeros((1, 3))),
            SourceRegion(np.zeros((1, 3)), np.array([1])),
        )
    with pytest.raises(ValueError, match="surface mesh and volume mesh dimensions"):
        TorsoGeometry(
            "bad",
            mesh,
            ElectrodeSet(np.zeros((1, 3))),
            source_region,
            surface_mesh=MeshData(points=np.zeros((2, 2)), cells=np.array([[0, 1]]), cell_type="line"),
        )


def test_affine_transform_factories_and_component_transforms():
    transform = AffineTransform(matrix=np.eye(3), offset=np.array([1.0, 2.0, 3.0]))
    mesh = single_tetra_mesh(name="mesh")
    electrodes = ElectrodeSet(np.array([[0.0, 0.0, 0.0]]), labels=["A"], metadata={"kind": "electrode"})
    region = SourceRegion.from_cell_ids(mesh, np.array([0]), metadata={"kind": "source"})

    assert np.allclose(AffineTransform.identity(3).apply_points([[1.0, 2.0, 3.0]]), [[1.0, 2.0, 3.0]])
    assert np.allclose(AffineTransform.scale(2.0, dim=3).matrix, np.diag([2.0, 2.0, 2.0]))
    assert np.allclose(AffineTransform.scale([2.0, 3.0, 4.0]).matrix, np.diag([2.0, 3.0, 4.0]))

    moved_mesh = transform_mesh(mesh, transform)
    moved_electrodes = transform_electrodes(electrodes, transform)
    moved_region = transform_source_region(region, transform)

    assert moved_mesh.name == "mesh_affine"
    assert np.allclose(moved_mesh.points[0], [1.0, 2.0, 3.0])
    assert moved_mesh.metadata["transform"] == "affine"
    assert np.allclose(moved_electrodes.positions, [[1.0, 2.0, 3.0]])
    assert moved_electrodes.metadata == {"kind": "electrode", "transform": "affine"}
    assert np.allclose(moved_region.candidate_points, [[1.25, 2.25, 3.25]])
    assert moved_region.metadata == {"kind": "source", "transform": "affine"}


def test_transform_torso_geometry_preserves_structure_and_records_registration():
    geom = simple_geometry()
    transform = AffineTransform(matrix=np.diag([2.0, 3.0, 4.0]), offset=np.array([1.0, 1.0, 1.0]))

    moved = transform_torso_geometry(geom, transform, geometry_id="moved")

    assert moved.geometry_id == "moved"
    assert moved.metadata["parent_geometry_id"] == "geom"
    assert moved.registration_transform["type"] == "affine"
    assert np.allclose(moved.volume_mesh.points[1], [3.0, 1.0, 1.0])
    assert np.allclose(moved.surface_mesh.points[2], [1.0, 4.0, 1.0])
    assert np.allclose(moved.electrodes.positions[1], [3.0, 1.0, 1.0])
    assert np.array_equal(moved.source_region.candidate_cell_ids, geom.source_region.candidate_cell_ids)


def test_affine_transform_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="matrix must be square"):
        AffineTransform(np.zeros((2, 3)), np.zeros(2))
    with pytest.raises(ValueError, match="offset must have shape"):
        AffineTransform(np.eye(3), np.zeros(2))
    with pytest.raises(ValueError, match="scale factors"):
        AffineTransform.scale([1.0, 2.0], dim=3)


def test_validate_torso_geometry_reports_errors_warnings_and_distances():
    mesh = MeshData(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ),
        cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
        cell_type="tetra",
    )
    geometry_with_empty_parts = TorsoGeometry(
        "invalid",
        mesh,
        ElectrodeSet(np.empty((0, 3))),
        SourceRegion(np.empty((0, 3)), np.empty((0,), dtype=np.int64)),
    )

    report = validate_torso_geometry(geometry_with_empty_parts)
    assert not report.is_valid
    assert "volume mesh has 1 degenerate cells" in report.errors
    assert "no electrodes provided" in report.errors
    assert "source region has no candidate points" in report.errors
    assert report.summary["mean_electrode_distance_to_nearest_node"] is None
    assert report.summary["max_electrode_distance_to_nearest_node"] is None

    far = TorsoGeometry(
        "far",
        single_tetra_mesh(),
        ElectrodeSet(np.array([[10.0, 0.0, 0.0]])),
        SourceRegion.from_cell_ids(single_tetra_mesh(), np.array([0])),
    )
    far_report = validate_torso_geometry(far, electrode_node_distance_warning=1.0)
    assert far_report.is_valid
    assert far_report.warnings == ["some electrodes are far from nearest mesh nodes: max distance=9"]
    assert far_report.summary["max_electrode_distance_to_nearest_node"] == pytest.approx(9.0)


def test_visualization_plots_mesh_source_region_and_geometry(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    geom = simple_geometry()

    mesh_path = tmp_path / "mesh.png"
    source_path = tmp_path / "source.png"
    geom_path = tmp_path / "geometry.png"

    fig1, ax1 = plot_mesh(geom.volume_mesh, show_points=True, save_path=mesh_path)
    fig2, ax2 = plot_source_region(geom.source_region, save_path=source_path)
    fig3, ax3 = plot_torso_geometry(geom, save_path=geom_path)

    assert ax1.get_title() == "single_tet"
    assert ax2.get_title() == "source_region"
    assert ax3.get_title() == "geom"
    assert mesh_path.exists()
    assert source_path.exists()
    assert geom_path.exists()

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
