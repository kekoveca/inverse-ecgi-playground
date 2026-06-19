from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from measurements import MeasurementOperator


@dataclass(frozen=True)
class GreenSolveInfo:
    """Diagnostics for one solve ``K G_i = M_i^T``."""

    row_index: int
    rhs_sum: float
    rhs_norm: float
    converged: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GreenBasis:
    """Green functions associated with rows of a measurement operator."""

    measurement_operator: MeasurementOperator
    functions: list[Any]
    reference: str
    sign: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.functions = list(self.functions)
        self.sign = float(self.sign)
        if not np.isfinite(self.sign) or self.sign == 0.0:
            raise ValueError("sign must be a finite nonzero number")

    @property
    def num_measurements(self) -> int:
        return int(self.measurement_operator.num_electrodes)

    @property
    def num_nodes(self) -> int:
        return int(self.measurement_operator.num_nodes)

    @property
    def has_functions(self) -> bool:
        return bool(self.functions)

    def function(self, index: int):
        """Return a stored Green function by basis-list index."""
        if not self.has_functions:
            raise ValueError("Green functions were not retained; construct GreenSolver with keep_functions=True")
        return self.functions[int(index)]

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary without function vectors."""
        return {
            "num_measurements": self.num_measurements,
            "num_nodes": self.num_nodes,
            "num_functions": len(self.functions),
            "has_functions": self.has_functions,
            "reference": self.reference,
            "sign": self.sign,
            "metadata": self.metadata,
        }
