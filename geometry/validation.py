from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .electrodes import electrode_placement_report
from .mesh_model import quality_report
from .torso_geometry import TorsoGeometry


@dataclass(frozen=True)
class GeometryValidationReport:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    summary: dict


def validate_torso_geometry(
    geometry: TorsoGeometry,
    min_cell_volume_eps: float = 1e-14,
    electrode_node_distance_warning: float | None = None,
) -> GeometryValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    mesh = geometry.volume_mesh
    q = quality_report(mesh, eps=min_cell_volume_eps)

    if q.num_cells == 0:
        errors.append("volume mesh has no cells")
    if q.num_points == 0:
        errors.append("volume mesh has no points")
    if q.num_degenerate_cells > 0:
        errors.append(f"volume mesh has {q.num_degenerate_cells} degenerate cells")

    if geometry.electrodes.num_electrodes == 0:
        errors.append("no electrodes provided")
    if geometry.source_region.num_candidates == 0:
        errors.append("source region has no candidate points")

    if np.any(~np.isfinite(mesh.points)):
        errors.append("mesh points contain non-finite values")
    if np.any(~np.isfinite(geometry.electrodes.positions)):
        errors.append("electrode positions contain non-finite values")
    if np.any(~np.isfinite(geometry.source_region.candidate_points)):
        errors.append("source candidate points contain non-finite values")

    if geometry.electrodes.num_electrodes > 0 and mesh.num_points > 0:
        placement = electrode_placement_report(geometry.electrodes, mesh)
        mean_distance_to_nearest_node = placement.mean_distance_to_nearest_node
        max_distance_to_nearest_node = placement.max_distance_to_nearest_node
    else:
        placement = None
        mean_distance_to_nearest_node = None
        max_distance_to_nearest_node = None

    if placement is not None and electrode_node_distance_warning is not None:
        if placement.max_distance_to_nearest_node > electrode_node_distance_warning:
            warnings.append(
                "some electrodes are far from nearest mesh nodes: "
                f"max distance={placement.max_distance_to_nearest_node:g}"
            )

    summary = geometry.summary()
    summary.update(
        {
            "mean_electrode_distance_to_nearest_node": mean_distance_to_nearest_node,
            "max_electrode_distance_to_nearest_node": max_distance_to_nearest_node,
        }
    )

    return GeometryValidationReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        summary=summary,
    )
