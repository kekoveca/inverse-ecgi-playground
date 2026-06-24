from __future__ import annotations

from inverse import SingleDipoleInverseSolver, inverse_reconstruction_metrics

from .inverse_results import InverseBenchmarkRecord, InverseBenchmarkResult
from .inverse_scenario import InverseBenchmarkScenario


class InverseBenchmarkRunner:
    """Apply single-dipole inverse reconstruction to forward benchmark records."""

    def __init__(self, inverse_solver_factory=None) -> None:
        if inverse_solver_factory is None:
            inverse_solver_factory = (
                lambda transfer, lambda_reg, reference: SingleDipoleInverseSolver(
                    transfer,
                    lambda_reg=lambda_reg,
                    reference=reference,
                )
            )
        if not callable(inverse_solver_factory):
            raise TypeError("inverse_solver_factory must be callable")
        self.inverse_solver_factory = inverse_solver_factory

    def _record_from_inverse(self, scenario, forward_record, inverse_result, measurement_kind: str):
        metrics = inverse_reconstruction_metrics(
            inverse_result,
            true_position=forward_record.source_position,
            true_moment=forward_record.source_moment,
            localization_threshold=scenario.localization_threshold,
        )
        return InverseBenchmarkRecord(
            scenario_name=scenario.name,
            source_index=forward_record.source_index,
            source_position=forward_record.source_position,
            source_moment=forward_record.source_moment,
            source_cell_id=forward_record.source_cell_id,
            electrode_set_name=forward_record.electrode_set_name,
            num_electrodes=forward_record.num_electrodes,
            noise_model_name=forward_record.noise_model_name,
            measurement_kind=measurement_kind,
            lambda_reg=scenario.lambda_reg,
            estimated_candidate_index=inverse_result.best_candidate_index,
            estimated_position=inverse_result.estimated_position,
            estimated_cell_id=inverse_result.estimated_cell_id,
            estimated_moment=inverse_result.estimated_moment,
            residual_norm=inverse_result.residual_norm,
            relative_residual=inverse_result.relative_residual,
            localization_error=metrics["localization_error"],
            moment_relative_error=metrics["moment_relative_error"],
            moment_angle_error_deg=metrics["moment_angle_error_deg"],
            success=metrics.get("success"),
            metadata={"inverse": inverse_result.to_summary_dict()},
        )

    def run(self, scenario: InverseBenchmarkScenario) -> InverseBenchmarkResult:
        scenario.validate()
        inverse_solver = self.inverse_solver_factory(
            scenario.transfer_matrix,
            scenario.lambda_reg,
            scenario.reference,
        )
        records = []
        for forward_record in scenario.records:
            if scenario.use_clean_measurements:
                inverse_result = inverse_solver.solve(forward_record.clean_measurements)
                records.append(self._record_from_inverse(scenario, forward_record, inverse_result, "clean"))
            if scenario.use_noisy_measurements:
                inverse_result = inverse_solver.solve(forward_record.noisy_measurements)
                records.append(self._record_from_inverse(scenario, forward_record, inverse_result, "noisy"))
        return InverseBenchmarkResult(
            scenario=scenario,
            records=records,
            metadata={"runner": self.__class__.__name__},
        )


def run_inverse_benchmark(
    forward_result,
    transfer_matrix,
    name: str = "inverse_benchmark",
    lambda_reg: float = 0.0,
    localization_threshold=None,
    use_clean_measurements: bool = True,
    use_noisy_measurements: bool = True,
    reference: str | None = None,
) -> InverseBenchmarkResult:
    """Convenience wrapper for inverse benchmark from forward records."""
    scenario = InverseBenchmarkScenario(
        name=name,
        forward_result=forward_result,
        transfer_matrix=transfer_matrix,
        lambda_reg=lambda_reg,
        localization_threshold=localization_threshold,
        use_clean_measurements=use_clean_measurements,
        use_noisy_measurements=use_noisy_measurements,
        reference=reference,
    )
    return InverseBenchmarkRunner().run(scenario)
