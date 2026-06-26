#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np


# Allow running as:
#   python3 examples/full_inverse_experiment_torso_clipped_sphere_electrodes.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import full_inverse_experiment_torso as base
from geometry import ElectrodeSet, electrode_placement_report
from measurements import central_project_electrodes_to_surface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full forward -> Green -> inverse experiment on torso.msh with electrodes "
            "generated on a clipped circumscribed sphere and centrally projected to the torso."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mesh", default="torso.msh", help="Path to Gmsh .msh torso mesh.")
    parser.add_argument(
        "--output",
        default="output/full_inverse_experiment_clipped_sphere_electrodes",
        help="Output directory.",
    )
    parser.add_argument("--domain-name", default="domain", help="Physical group name for tetra volume.")
    parser.add_argument("--boundary-name", default="boundary", help="Physical group name for triangle boundary.")
    parser.add_argument("--num-electrodes", type=int, default=32, help="Number of projected electrodes.")
    parser.add_argument("--num-candidates", type=int, default=50, help="Number of source candidates.")
    parser.add_argument(
        "--electrode-offset",
        type=float,
        default=0.0,
        help="Reserved for projection APIs with normal offsets; current central projection ignores it.",
    )
    parser.add_argument(
        "--electrode-mode",
        choices=["clipped-sphere-fibonacci"],
        default="clipped-sphere-fibonacci",
        help="Electrode layout before projection.",
    )
    parser.add_argument(
        "--z-trim-fraction",
        type=float,
        default=0.1,
        help="Move bbox z_min/z_max clipping planes this fraction toward the bbox center.",
    )
    parser.add_argument("--moment", nargs=3, type=float, default=[0.0, 0.0, 1.0], help="True dipole moment.")
    parser.add_argument("--source-index", type=int, default=None, help="True source candidate index.")
    parser.add_argument("--snr-db", type=float, default=None, help="Optional relative Gaussian noise SNR in dB.")
    parser.add_argument("--lambda-reg", type=float, default=1e-10, help="Tikhonov regularization for inverse LS.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Constant scalar conductivity.")
    parser.add_argument(
        "--reference",
        choices=["average", "single", "none"],
        default="average",
        help="Electrode reference system. Green solves usually require average or single.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the golden-angle phase.")
    parser.add_argument("--no-export", action="store_true", help="Skip VTX/BP ParaView exports.")
    parser.add_argument(
        "--green-sign-check",
        dest="green_sign_check",
        action="store_true",
        default=True,
        help="Run forward/Green sign consistency diagnostics.",
    )
    parser.add_argument(
        "--skip-green-sign-check",
        dest="green_sign_check",
        action="store_false",
        help="Skip forward/Green sign consistency diagnostics.",
    )
    return parser.parse_args()


def clipped_bbox_sphere_parameters(volume_mesh, z_trim_fraction: float = 0.1) -> dict[str, Any]:
    """Return sphere and z-clipping planes derived from the volume bbox.

    The sphere is centered at the bbox center and circumscribes the bbox. The
    lower/upper clipping planes start at bbox z_min/z_max and move 10% toward
    the center by default.
    """
    z_trim_fraction = float(z_trim_fraction)
    if not 0.0 <= z_trim_fraction < 0.5:
        raise ValueError("z_trim_fraction must satisfy 0 <= z_trim_fraction < 0.5")

    bbox_min, bbox_max = volume_mesh.bounding_box()
    center = 0.5 * (bbox_min + bbox_max)
    half_extents = 0.5 * (bbox_max - bbox_min)
    radius = float(np.linalg.norm(half_extents))
    if radius <= 0.0:
        raise ValueError("volume bbox is degenerate; cannot build circumscribed sphere")

    z_min = float(bbox_min[2] + z_trim_fraction * (center[2] - bbox_min[2]))
    z_max = float(bbox_max[2] - z_trim_fraction * (bbox_max[2] - center[2]))
    if not z_min < z_max:
        raise ValueError("z clipping planes collapsed; reduce z_trim_fraction")

    return {
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "center": center,
        "radius": radius,
        "z_clip_min": z_min,
        "z_clip_max": z_max,
        "z_trim_fraction": z_trim_fraction,
    }


def quasiuniform_points_on_clipped_sphere(
    center,
    radius: float,
    z_clip_min: float,
    z_clip_max: float,
    num_points: int,
    seed: int = 0,
) -> np.ndarray:
    """Generate quasi-uniform points on a z-clipped sphere using a golden angle."""
    center = np.asarray(center, dtype=float)
    radius = float(radius)
    num_points = int(num_points)
    if center.shape != (3,):
        raise ValueError("center must have shape (3,)")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if num_points < 1:
        raise ValueError("num_points must be positive")

    z_rel_min = max(float(z_clip_min) - center[2], -radius)
    z_rel_max = min(float(z_clip_max) - center[2], radius)
    if not z_rel_min < z_rel_max:
        raise ValueError("z clipping interval does not intersect the sphere")

    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    indices = np.arange(num_points, dtype=float)
    z_rel = z_rel_min + (indices + 0.5) * (z_rel_max - z_rel_min) / num_points
    xy_radius = np.sqrt(np.maximum(radius * radius - z_rel * z_rel, 0.0))
    theta = phase + indices * golden_angle

    points = np.column_stack(
        [
            center[0] + xy_radius * np.cos(theta),
            center[1] + xy_radius * np.sin(theta),
            center[2] + z_rel,
        ]
    )
    return points


def build_clipped_sphere_projected_electrodes(
    volume_mesh,
    surface_mesh,
    num_electrodes: int,
    mode: str,
    seed: int,
    offset: float,
    z_trim_fraction: float = 0.1,
) -> tuple[ElectrodeSet, dict[str, Any], dict[str, Any]]:
    """Generate clipped-sphere electrodes and project them to the torso surface."""
    if mode != "clipped-sphere-fibonacci":
        raise ValueError(f"unsupported electrode mode {mode!r}")
    num_electrodes = int(num_electrodes)
    if num_electrodes < 1:
        raise ValueError("--num-electrodes must be positive")
    if offset != 0.0:
        warnings.warn(
            "--electrode-offset was provided, but the current production central projection API "
            "does not support normal offsets; the offset is not applied.",
            RuntimeWarning,
        )

    sphere = clipped_bbox_sphere_parameters(volume_mesh, z_trim_fraction=z_trim_fraction)
    sphere_positions = quasiuniform_points_on_clipped_sphere(
        center=sphere["center"],
        radius=sphere["radius"],
        z_clip_min=sphere["z_clip_min"],
        z_clip_max=sphere["z_clip_max"],
        num_points=num_electrodes,
        seed=seed,
    )
    initial_electrodes = ElectrodeSet(
        positions=sphere_positions,
        labels=[f"SPH{i + 1:03d}" for i in range(num_electrodes)],
        name="clipped_sphere_electrodes",
        metadata={
            "source": "bbox_circumscribed_clipped_sphere",
            "sphere": {
                "center": sphere["center"].tolist(),
                "radius": sphere["radius"],
                "z_clip_min": sphere["z_clip_min"],
                "z_clip_max": sphere["z_clip_max"],
                "z_trim_fraction": sphere["z_trim_fraction"],
            },
        },
    )

    projected_electrodes, projection_report = central_project_electrodes_to_surface(
        volume_mesh=volume_mesh,
        electrodes=initial_electrodes,
        surface_mesh=surface_mesh,
        center=sphere["center"],
    )
    projection_summary = projection_report.to_summary_dict()
    projection_summary.update(
        {
            "selection_mode": mode,
            "production_projection_api": "measurements.central_project_electrodes_to_surface",
            "source_layout": "bbox_circumscribed_clipped_sphere",
            "sphere_center": sphere["center"],
            "sphere_radius": sphere["radius"],
            "sphere_bbox_min": sphere["bbox_min"],
            "sphere_bbox_max": sphere["bbox_max"],
            "sphere_z_clip_min": sphere["z_clip_min"],
            "sphere_z_clip_max": sphere["z_clip_max"],
            "sphere_z_trim_fraction": sphere["z_trim_fraction"],
            "offset_requested": float(offset),
            "offset_applied": False,
            "num_unchanged": int(projected_electrodes.num_electrodes - projection_report.num_projected),
            "num_projection_surface_cell_ids_missing": int(np.count_nonzero(projection_report.surface_cell_ids < 0)),
        }
    )
    surface_diagnostics = base.inspect_electrodes_nearest_surface_cells(surface_mesh, projected_electrodes)

    print("Clipped sphere electrode source:")
    print(f"  sphere center: {sphere['center']}")
    print(f"  sphere radius: {sphere['radius']:.6g}")
    print(f"  z clip min/max: {sphere['z_clip_min']:.6g}, {sphere['z_clip_max']:.6g}")
    print(f"  z trim fraction: {sphere['z_trim_fraction']:.6g}")
    print(f"Electrodes: {projected_electrodes.num_electrodes}")
    print("Projection:")
    print(f"  production API: {projection_summary['production_projection_api']}")
    print(f"  only_outside_electrodes: {projection_summary['metadata'].get('only_outside_electrodes')}")
    print(f"  num projected: {projection_summary['num_projected']}")
    print(f"  num unchanged: {projection_summary['num_unchanged']}")
    print(f"  offset requested: {projection_summary['offset_requested']}")
    print(f"  offset applied: {projection_summary['offset_applied']}")
    print("First projected electrodes:")
    for label, position in zip(projected_electrodes.labels[:5], projected_electrodes.positions[:5]):
        print(f"  {label}: {position}")

    placement = electrode_placement_report(projected_electrodes, surface_mesh)
    electrodes_on_surface_nodes = bool(placement.max_distance_to_nearest_node <= 1e-10)
    projection_summary["electrodes_exactly_on_surface_nodes"] = electrodes_on_surface_nodes
    print(
        "Electrode nearest-surface-node distance: "
        f"mean={placement.mean_distance_to_nearest_node:.6g}, "
        f"max={placement.max_distance_to_nearest_node:.6g}"
    )
    print("Surface location:")
    print(
        "  nearest surface cell ids available: "
        f"{'yes' if surface_diagnostics['num_nearest_surface_cell_ids_missing'] == 0 else 'no'}"
    )
    print(f"  nearest surface distance mean: {surface_diagnostics['mean_nearest_surface_distance']:.6g}")
    print(f"  nearest surface distance max: {surface_diagnostics['max_nearest_surface_distance']:.6g}")
    print(f"  electrodes exactly on surface nodes: {'yes' if electrodes_on_surface_nodes else 'no'}")
    return projected_electrodes, projection_summary, surface_diagnostics


def main() -> int:
    args = parse_args()

    def electrode_builder(volume_mesh, surface_mesh, num_electrodes, mode, seed, offset):
        return build_clipped_sphere_projected_electrodes(
            volume_mesh=volume_mesh,
            surface_mesh=surface_mesh,
            num_electrodes=num_electrodes,
            mode=mode,
            seed=seed,
            offset=offset,
            z_trim_fraction=args.z_trim_fraction,
        )

    return base.run_full_inverse_experiment(args, electrode_builder=electrode_builder)


if __name__ == "__main__":
    raise SystemExit(main())
