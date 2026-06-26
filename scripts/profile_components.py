#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from green import GreenTransferMatrix
from inverse import SingleDipoleInverseSolver
from performance import PerformanceTimer, format_timing_table, save_timing_csv, save_timing_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile isolated performance-sensitive components.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--component", choices=["inverse-scaling"], default="inverse-scaling")
    parser.add_argument("--output", default="output/component_profile")
    parser.add_argument("--candidate-counts", nargs="+", type=int, default=[100, 1000, 10000])
    parser.add_argument("--measurement-counts", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--lambda-reg", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def profile_inverse_scaling(args: argparse.Namespace) -> PerformanceTimer:
    timer = PerformanceTimer()
    rng = np.random.default_rng(args.seed)
    p_true = np.array([0.7, -1.2, 0.5])
    for num_measurements in args.measurement_counts:
        for num_candidates in args.candidate_counts:
            A = rng.normal(size=(num_candidates, num_measurements, 3))
            true_index = min(num_candidates // 3, num_candidates - 1)
            candidate_points = rng.normal(size=(num_candidates, 3))
            candidate_cell_ids = np.arange(num_candidates, dtype=np.int64)
            transfer = GreenTransferMatrix(
                A=A,
                candidate_points=candidate_points,
                candidate_cell_ids=candidate_cell_ids,
            )
            measurements = transfer.matrix_for_candidate(true_index) @ p_true
            solver = SingleDipoleInverseSolver(transfer, lambda_reg=args.lambda_reg)
            with timer.time(
                "inverse_solve",
                num_candidates=int(num_candidates),
                num_measurements=int(num_measurements),
            ):
                result = solver.solve(measurements)
            timer.records[-1].metadata["best_candidate"] = int(result.best_candidate_index)
            timer.records[-1].metadata["relative_residual"] = float(result.relative_residual)
    return timer


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.component == "inverse-scaling":
        timer = profile_inverse_scaling(args)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(args.component)

    save_timing_csv(timer, output_dir / "timing.csv")
    save_timing_json(timer, output_dir / "timing.json")
    (output_dir / "profile_summary.md").write_text(
        "# Component Performance Profile\n\n" + format_timing_table(timer) + "\n",
        encoding="utf-8",
    )
    print(format_timing_table(timer))
    print(f"Saved component profile to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
