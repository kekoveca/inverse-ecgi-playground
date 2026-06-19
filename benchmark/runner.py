from __future__ import annotations

from pathlib import Path

import numpy as np

from forward import ForwardSolver, export_forward_result_to_vtx

from .metrics import noise_metrics
from .results import ForwardBenchmarkRecord, ForwardBenchmarkResult
from .scenario import ForwardBenchmarkScenario


class ForwardBenchmarkRunner:
    """Execute forward-only source/electrode/noise benchmark scenarios."""

    def __init__(
        self,
        poisson_solver_factory,
        export_potentials: bool = False,
        output_dir=None,
        keep_potentials: bool = False,
    ) -> None:
        if not callable(poisson_solver_factory):
            raise TypeError("poisson_solver_factory must be callable")
        if export_potentials and output_dir is None:
            raise ValueError("output_dir is required when export_potentials=True")
        if keep_potentials:
            raise NotImplementedError("keeping potentials in benchmark results is not implemented")
        self.poisson_solver_factory = poisson_solver_factory
        self.export_potentials = bool(export_potentials)
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.keep_potentials = False

    def run(self, scenario: ForwardBenchmarkScenario) -> ForwardBenchmarkResult:
        scenario.validate()
        solver = self.poisson_solver_factory(scenario.geometry.volume_mesh)
        records = []
        rngs = [np.random.default_rng(getattr(model, "seed", None)) for model in scenario.noise_models]
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            for electrode_subset in scenario.electrode_sets:
                forward = ForwardSolver(
                    poisson_solver=solver,
                    electrodes=electrode_subset.electrodes,
                    reference=scenario.reference,
                    reference_index=scenario.reference_index,
                )
                for source_index, source in enumerate(scenario.sources):
                    forward_result = forward.solve(source)
                    clean = np.asarray(forward_result.measurements, dtype=float).copy()

                    potential_path = None
                    if self.export_potentials:
                        filename = f"potential_{electrode_subset.name}_source_{source_index:06d}.bp"
                        potential_path = export_forward_result_to_vtx(forward_result, self.output_dir / filename)

                    for model_index, noise_model in enumerate(scenario.noise_models):
                        noisy, noise = noise_model.apply(clean, rng=rngs[model_index])
                        metrics = noise_metrics(clean, noisy, noise)
                        metadata = {
                            "forward": dict(forward_result.metadata),
                            "electrode_subset": electrode_subset.to_config_dict(),
                        }
                        if potential_path is not None:
                            metadata["potential_path"] = str(potential_path)
                        records.append(
                            ForwardBenchmarkRecord(
                                scenario_name=scenario.name,
                                source_index=source_index,
                                source_position=source.position,
                                source_moment=source.moment,
                                source_cell_id=source.cell_id,
                                electrode_set_name=electrode_subset.name,
                                num_electrodes=clean.size,
                                noise_model_name=noise_model.name,
                                reference=scenario.reference,
                                clean_measurements=clean,
                                noisy_measurements=noisy,
                                noise=noise,
                                metrics=metrics,
                                metadata=metadata,
                            )
                        )
            return ForwardBenchmarkResult(
                scenario=scenario,
                records=records,
                metadata={"runner": self.__class__.__name__, "potentials_exported": self.export_potentials},
            )
        finally:
            destroy = getattr(solver, "destroy", None)
            if callable(destroy):
                destroy()


def run_forward_benchmark(scenario, poisson_solver_factory, **kwargs) -> ForwardBenchmarkResult:
    """Run a scenario with a newly constructed ``ForwardBenchmarkRunner``."""
    return ForwardBenchmarkRunner(poisson_solver_factory, **kwargs).run(scenario)
