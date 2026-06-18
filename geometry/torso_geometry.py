from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .electrodes import ElectrodeSet
from .mesh_model import MeshData, MeshQualityReport, quality_report
from .source_region import SourceRegion


@dataclass(frozen=True)
class TorsoGeometry:
    """Validated geometry bundle for forward and future inverse workflows.

    It groups the volume mesh, optional surface mesh, electrodes and source
    region without introducing a dependency on FEniCSx.
    """

    geometry_id: str
    volume_mesh: MeshData
    electrodes: ElectrodeSet
    source_region: SourceRegion
    surface_mesh: MeshData | None = None
    registration_transform: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dim = self.volume_mesh.geometric_dim
        if self.electrodes.geometric_dim != dim:
            raise ValueError("electrodes and volume mesh dimensions do not match")
        if self.source_region.geometric_dim != dim:
            raise ValueError("source region and volume mesh dimensions do not match")
        if self.source_region.candidate_cell_ids.size > 0:
            if self.source_region.candidate_cell_ids.min() < 0:
                raise ValueError("source region has negative cell ids")
            if self.source_region.candidate_cell_ids.max() >= self.volume_mesh.num_cells:
                raise ValueError("source region has cell ids outside volume mesh")
        if self.surface_mesh is not None and self.surface_mesh.geometric_dim != dim:
            raise ValueError("surface mesh and volume mesh dimensions do not match")

    def quality_report(self) -> MeshQualityReport:
        return quality_report(self.volume_mesh)

    def summary(self) -> dict[str, Any]:
        q = self.quality_report()
        return {
            "geometry_id": self.geometry_id,
            "num_volume_points": self.volume_mesh.num_points,
            "num_volume_cells": self.volume_mesh.num_cells,
            "num_electrodes": self.electrodes.num_electrodes,
            "num_source_candidates": self.source_region.num_candidates,
            "geometric_dim": self.volume_mesh.geometric_dim,
            "min_cell_volume": q.min_cell_volume,
            "num_degenerate_cells": q.num_degenerate_cells,
        }
