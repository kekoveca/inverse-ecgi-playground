from __future__ import annotations

import numpy as np

from geometry import ElectrodeSet, MeshData

from .measurement_operator import MeasurementOperator, build_measurement_operator


def measure_nodal_values(
    mesh: MeshData,
    electrodes: ElectrodeSet,
    nodal_values,
    reference: str = "average",
    reference_index: int | None = None,
    tol: float = 1e-10,
    surface_mesh: MeshData | None = None,
    project_outside_electrodes: bool = True,
    projection_center=None,
) -> np.ndarray:
    """Evaluate nodal values at electrodes and apply the requested reference."""
    op = build_measurement_operator(
        mesh,
        electrodes,
        reference=reference,
        reference_index=reference_index,
        sparse=True,
        tol=tol,
        surface_mesh=surface_mesh,
        project_outside_electrodes=project_outside_electrodes,
        projection_center=projection_center,
    )
    return op.evaluate(nodal_values)


def measure_raw_nodal_values(
    mesh: MeshData,
    electrodes: ElectrodeSet,
    nodal_values,
    tol: float = 1e-10,
    surface_mesh: MeshData | None = None,
    project_outside_electrodes: bool = True,
    projection_center=None,
) -> np.ndarray:
    """Evaluate nodal values at electrodes without re-referencing."""
    op = build_measurement_operator(
        mesh,
        electrodes,
        reference="none",
        sparse=True,
        tol=tol,
        surface_mesh=surface_mesh,
        project_outside_electrodes=project_outside_electrodes,
        projection_center=projection_center,
    )
    return op.evaluate_raw(nodal_values)


def measure_fenics_function(function, measurement_operator: MeasurementOperator) -> np.ndarray:
    """Evaluate a DOLFINx Function with a prebuilt measurement operator."""
    if not hasattr(function, "x") or not hasattr(function.x, "array"):
        raise NotImplementedError("measure_fenics_function expects a dolfinx.fem.Function-like object with x.array")
    return measurement_operator.evaluate(np.asarray(function.x.array, dtype=float))
