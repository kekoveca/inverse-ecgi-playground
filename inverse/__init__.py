"""Single-dipole inverse reconstruction from Green transfer matrices."""

from .diagnostics import format_inverse_summary, summarize_inverse_result
from .metrics import (
    inverse_reconstruction_metrics,
    is_successful_localization,
    localization_error,
    moment_angle_error_deg,
    moment_relative_error,
)
from .regularization import (
    condition_number,
    relative_residual,
    residual_norm,
    residual_vector,
    solve_tikhonov_moment,
)
from .result import CandidateInverseSolution, SingleDipoleInverseResult
from .single_dipole import SingleDipoleInverseSolver, solve_single_dipole_inverse

__all__ = [
    "CandidateInverseSolution",
    "SingleDipoleInverseResult",
    "SingleDipoleInverseSolver",
    "condition_number",
    "format_inverse_summary",
    "inverse_reconstruction_metrics",
    "is_successful_localization",
    "localization_error",
    "moment_angle_error_deg",
    "moment_relative_error",
    "relative_residual",
    "residual_norm",
    "residual_vector",
    "solve_single_dipole_inverse",
    "solve_tikhonov_moment",
    "summarize_inverse_result",
]
