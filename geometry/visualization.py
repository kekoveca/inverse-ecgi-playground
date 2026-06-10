from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .mesh_model import MeshData
from .source_region import SourceRegion
from .torso_geometry import TorsoGeometry


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Geometry visualization requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc
    return plt


def _set_equal_aspect_3d(ax: Any, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius == 0.0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_mesh(
    mesh: MeshData,
    ax: Any | None = None,
    show_points: bool = False,
    max_cells: int | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """Plot a lightweight MeshData object.

    Supports 2D line/triangle meshes and simple 3D scatter/edge rendering for
    tetra/triangle meshes. This is intended for diagnostics, not publication
    figures.
    """
    plt = _require_matplotlib()

    if ax is None:
        if mesh.geometric_dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots()
    else:
        fig = ax.figure

    points = mesh.points
    cells = mesh.cells if max_cells is None else mesh.cells[:max_cells]

    if mesh.geometric_dim == 2:
        if mesh.cell_type in {"triangle", "line"}:
            for cell in cells:
                poly = points[cell]
                if mesh.cell_type == "triangle":
                    poly = np.vstack([poly, poly[0]])
                ax.plot(poly[:, 0], poly[:, 1], linewidth=0.8)
        if show_points:
            ax.scatter(points[:, 0], points[:, 1], s=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    elif mesh.geometric_dim == 3:
        if mesh.cell_type == "tetra":
            edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        elif mesh.cell_type == "triangle":
            edge_pairs = [(0, 1), (1, 2), (2, 0)]
        elif mesh.cell_type == "line":
            edge_pairs = [(0, 1)]
        else:
            edge_pairs = []

        for cell in cells:
            verts = points[cell]
            for i, j in edge_pairs:
                seg = verts[[i, j]]
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=0.5)
        if show_points:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=8)
        _set_equal_aspect_3d(ax, points)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(mesh.name)

    if save_path is not None:
        fig.savefig(Path(save_path), bbox_inches="tight", dpi=200)

    return fig, ax


def plot_source_region(
    source_region: SourceRegion,
    ax: Any | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """Plot candidate source points."""
    plt = _require_matplotlib()
    points = source_region.candidate_points

    if ax is None:
        if source_region.geometric_dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if source_region.geometric_dim == 2:
        ax.scatter(points[:, 0], points[:, 1], s=20)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=20)
        _set_equal_aspect_3d(ax, points)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    ax.set_title(title or source_region.name)
    if save_path is not None:
        fig.savefig(Path(save_path), bbox_inches="tight", dpi=200)
    return fig, ax


def plot_torso_geometry(
    geometry: TorsoGeometry,
    ax: Any | None = None,
    show_volume_mesh: bool = True,
    show_surface_mesh: bool = False,
    show_electrodes: bool = True,
    show_source_region: bool = True,
    max_cells: int | None = 500,
    title: str | None = None,
    save_path: str | Path | None = None,
    show_fig: bool = False,
):
    """Plot a TorsoGeometry diagnostic view.

    For large 3D meshes use ``max_cells`` to keep plotting responsive. The
    function intentionally avoids advanced rendering libraries so it remains a
    lightweight optional diagnostic helper.
    """
    plt = _require_matplotlib()

    mesh = geometry.surface_mesh if show_surface_mesh and geometry.surface_mesh is not None else geometry.volume_mesh
    if ax is None:
        if geometry.volume_mesh.geometric_dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if show_volume_mesh:
        plot_mesh(mesh, ax=ax, show_points=False, max_cells=max_cells, title=None)

    dim = geometry.volume_mesh.geometric_dim
    if show_electrodes and geometry.electrodes.num_electrodes > 0:
        e = geometry.electrodes.positions
        if dim == 2:
            ax.scatter(e[:, 0], e[:, 1], s=50, marker="x", label="electrodes")
            for label, pos in zip(geometry.electrodes.labels, e):
                ax.text(pos[0], pos[1], label)
        else:
            ax.scatter(e[:, 0], e[:, 1], e[:, 2], s=50, marker="x", label="electrodes")
            for label, pos in zip(geometry.electrodes.labels, e):
                ax.text(pos[0], pos[1], pos[2], label)

    if show_source_region and geometry.source_region.num_candidates > 0:
        s = geometry.source_region.candidate_points
        if dim == 2:
            ax.scatter(s[:, 0], s[:, 1], s=18, marker="o", label="source candidates")
        else:
            ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=18, marker="o", label="source candidates")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(geometry.geometry_id)

    if show_electrodes or show_source_region:
        ax.legend(loc="best")

    if save_path is not None:
        fig.savefig(Path(save_path), bbox_inches="tight", dpi=200)

    if show_fig:
        import matplotlib.pyplot as plt

        plt.show()

    return fig, ax
