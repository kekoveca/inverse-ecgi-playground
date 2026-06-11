from __future__ import annotations

import numpy as np

from geometry import ElectrodeSet, MeshData

from .interpolation import evaluate_at_points
from .measurement_operator import MeasurementOperator, build_measurement_operator


def measure_nodal_values(
    mesh: MeshData,
    electrodes: ElectrodeSet,
    nodal_values,
    reference: str = "average",
    reference_index: int | None = None,
    tol: float = 1e-10,
) -> np.ndarray:
    """Evaluate nodal values at electrodes and apply the requested reference."""
    op = build_measurement_operator(
        mesh,
        electrodes,
        reference=reference,
        reference_index=reference_index,
        sparse=True,
        tol=tol,
    )
    return op.evaluate(nodal_values)


def measure_raw_nodal_values(
    mesh: MeshData,
    electrodes: ElectrodeSet,
    nodal_values,
    tol: float = 1e-10,
) -> np.ndarray:
    """Evaluate nodal values at electrodes without re-referencing."""
    return evaluate_at_points(mesh, nodal_values, electrodes.positions, tol=tol)


def measure_fenics_function(function, measurement_operator: MeasurementOperator) -> np.ndarray:
    """Evaluate a DOLFINx Function with a prebuilt measurement operator."""
    if not hasattr(function, "x") or not hasattr(function.x, "array"):
        raise NotImplementedError("measure_fenics_function expects a dolfinx.fem.Function-like object with x.array")
    return measurement_operator.evaluate(np.asarray(function.x.array, dtype=float))
