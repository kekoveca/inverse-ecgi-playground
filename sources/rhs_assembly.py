from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from geometry import MeshData

from .cell_geometry import barycentric_coordinates_tetra, gradients_p1_tetra, point_in_tetra
from .point_dipole import PointDipole


def _validate_tetra_mesh(mesh: MeshData) -> None:
    if mesh.cell_type != "tetra":
        raise ValueError("point dipole RHS assembly requires mesh.cell_type='tetra'")
    if mesh.geometric_dim != 3:
        raise ValueError("point dipole RHS assembly requires 3D mesh points")
    if mesh.cells.shape[1] != 4:
        raise ValueError("tetra mesh cells must have shape (n_cells, 4)")


def locate_point_in_mesh(
    mesh: MeshData,
    point: np.ndarray,
    candidate_cell_ids=None,
    tol: float = 1e-10,
) -> int:
    """Return the first tetrahedron id containing ``point``."""
    return int(locate_points_in_mesh(mesh, np.asarray(point, dtype=float).reshape(1, -1), candidate_cell_ids, tol)[0])


def locate_points_in_mesh(
    mesh: MeshData,
    points: np.ndarray,
    candidate_cell_ids=None,
    tol: float = 1e-10,
    initial_k: int = 8,
) -> np.ndarray:
    """Locate points in tetrahedral cells using a cKDTree over cell centroids.

    The KD-tree is used only to order likely cells. A point is accepted only
    after the barycentric ``point_in_tetra`` check succeeds, so the result is
    still geometric rather than nearest-centroid based.
    """
    _validate_tetra_mesh(mesh)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (n_points, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    if candidate_cell_ids is None:
        cell_ids = np.arange(mesh.num_cells, dtype=np.int64)
    else:
        cell_ids = np.asarray(candidate_cell_ids, dtype=np.int64)
        if cell_ids.ndim != 1:
            raise ValueError("candidate_cell_ids must be one-dimensional")
    if cell_ids.size == 0:
        raise ValueError("candidate_cell_ids must contain at least one cell")
    if cell_ids.min() < 0 or cell_ids.max() >= mesh.num_cells:
        raise ValueError("candidate_cell_ids contain ids outside mesh cells")

    if initial_k < 1:
        raise ValueError("initial_k must be positive")

    centroids = mesh.points[mesh.cells[cell_ids]].mean(axis=1)
    tree = cKDTree(centroids)
    located = np.full(points.shape[0], -1, dtype=np.int64)

    k = min(int(initial_k), cell_ids.size)
    while np.any(located < 0):
        unresolved = np.flatnonzero(located < 0)
        _, local_indices = tree.query(points[unresolved], k=k)
        if k == 1:
            local_indices = local_indices[:, np.newaxis]

        for row, point_id in enumerate(unresolved):
            point = points[point_id]
            for local_cell_id in np.atleast_1d(local_indices[row]):
                cell_id = int(cell_ids[int(local_cell_id)])
                vertices = mesh.points[mesh.cells[cell_id]]
                if point_in_tetra(point, vertices, tol=tol):
                    located[point_id] = cell_id
                    break

        if not np.any(located < 0):
            break
        if k == cell_ids.size:
            first_missing = int(np.flatnonzero(located < 0)[0])
            raise ValueError(f"point {first_missing} is not inside any candidate tetrahedron")
        k = min(2 * k, cell_ids.size)

    return located


def _resolve_cell_id(mesh: MeshData, source: PointDipole, cell_id: int | None) -> int:
    if cell_id is not None:
        resolved = int(cell_id)
    elif source.cell_id is not None:
        resolved = int(source.cell_id)
    else:
        resolved = locate_point_in_mesh(mesh, source.position)
    if resolved < 0 or resolved >= mesh.num_cells:
        raise ValueError(f"cell_id {resolved} is outside mesh cells")
    return resolved


def assemble_point_dipole_rhs_numpy(mesh: MeshData, source: PointDipole, cell_id: int | None = None) -> np.ndarray:
    """Assemble the nodal RHS for a P1 point dipole on a tetrahedral mesh."""
    _validate_tetra_mesh(mesh)
    resolved_cell_id = _resolve_cell_id(mesh, source, cell_id)

    global_dofs = mesh.cells[resolved_cell_id]
    vertices = mesh.points[global_dofs]
    grads = gradients_p1_tetra(vertices)
    local_rhs = grads @ source.moment

    rhs = np.zeros(mesh.num_points, dtype=float)
    np.add.at(rhs, global_dofs, local_rhs)
    return rhs


def rhs_compatibility_error(rhs) -> float:
    rhs = np.asarray(rhs, dtype=float)
    return abs(float(rhs.sum()))


def check_rhs_compatibility(rhs, tol: float = 1e-10) -> bool:
    rhs = np.asarray(rhs, dtype=float)
    norm = float(np.linalg.norm(rhs))
    scale = max(1.0, norm)
    return rhs_compatibility_error(rhs) <= tol * scale


def _solver_mesh_data(solver) -> MeshData:
    for name in ("mesh_data", "input_mesh", "mesh"):
        mesh = getattr(solver, name, None)
        if isinstance(mesh, MeshData):
            return mesh
    raise TypeError("solver must expose MeshData as mesh_data, input_mesh, or mesh")


def _solver_function_space(solver):
    V = getattr(solver, "V", None)
    if V is None:
        raise TypeError("solver must expose a DOLFINx FunctionSpace as V")
    return V


def _num_local_dolfinx_cells(solver) -> int:
    domain = getattr(solver, "domain", None)
    if domain is None:
        raise TypeError("solver must expose a DOLFINx mesh as domain")
    tdim = int(domain.topology.dim)
    index_map = domain.topology.index_map(tdim)
    if index_map is None:
        raise RuntimeError("DOLFINx cell index map is not available")
    return int(index_map.size_local)


def _dolfinx_cell_geometry(solver, cell_id: int) -> tuple[np.ndarray, np.ndarray]:
    V = _solver_function_space(solver)
    cell_id = int(cell_id)
    num_cells = _num_local_dolfinx_cells(solver)
    if cell_id < 0 or cell_id >= num_cells:
        raise ValueError(f"DOLFINx cell_id {cell_id} is outside local cells [0, {num_cells})")

    cell_dofs = np.asarray(V.dofmap.cell_dofs(cell_id), dtype=np.int64)
    if cell_dofs.shape != (4,):
        raise NotImplementedError("Point dipole RHS assembly currently supports only scalar P1 tetra spaces.")
    dof_coords = np.asarray(V.tabulate_dof_coordinates(), dtype=float)
    return cell_dofs, dof_coords[cell_dofs, :3].copy()


def locate_point_in_dolfinx_p1_tetra_mesh(
    solver,
    point,
    candidate_cell_ids=None,
    tol: float = 1e-10,
) -> int:
    """Return the local DOLFINx cell id containing ``point``."""
    point = np.asarray(point, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"point must have shape (3,), got {point.shape}")
    if not np.all(np.isfinite(point)):
        raise ValueError("point must contain only finite values")

    V = _solver_function_space(solver)
    num_cells = _num_local_dolfinx_cells(solver)
    if candidate_cell_ids is None:
        cell_ids = np.arange(num_cells, dtype=np.int64)
    else:
        cell_ids = np.asarray(candidate_cell_ids, dtype=np.int64)
        if cell_ids.ndim != 1:
            raise ValueError("candidate_cell_ids must be one-dimensional")
        if cell_ids.size == 0:
            raise ValueError("candidate_cell_ids must contain at least one cell")
        if cell_ids.min() < 0 or cell_ids.max() >= num_cells:
            raise ValueError("candidate_cell_ids contain ids outside local DOLFINx cells")

    dof_coords = np.asarray(V.tabulate_dof_coordinates(), dtype=float)
    for candidate in cell_ids:
        candidate_int = int(candidate)
        cell_dofs = np.asarray(V.dofmap.cell_dofs(candidate_int), dtype=np.int64)
        if cell_dofs.shape != (4,):
            raise NotImplementedError("Point dipole location currently supports only scalar P1 tetra spaces.")
        vertices = dof_coords[cell_dofs, :3]
        if point_in_tetra(point, vertices, tol=tol):
            return candidate_int
    raise ValueError(f"point {point.tolist()} is not inside any local DOLFINx P1 tetra cell")


def _resolve_dolfinx_cell_id(
    solver,
    source: PointDipole,
    cell_id: int | None,
    *,
    trust_source_cell_id: bool,
    tol: float,
) -> int:
    if cell_id is not None:
        return int(cell_id)
    if trust_source_cell_id and source.cell_id is not None:
        return int(source.cell_id)
    return locate_point_in_dolfinx_p1_tetra_mesh(solver, source.position, tol=tol)


def _point_dipole_local_rhs_in_dof_order(
    solver,
    source: PointDipole,
    cell_id: int | None = None,
    trust_source_cell_id: bool = False,
    tol: float = 1e-10,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    mesh_data = _solver_mesh_data(solver)
    _validate_tetra_mesh(mesh_data)
    resolved_cell_id = _resolve_dolfinx_cell_id(
        solver,
        source,
        cell_id,
        trust_source_cell_id=trust_source_cell_id,
        tol=tol,
    )
    cell_dofs, vertices = _dolfinx_cell_geometry(solver, resolved_cell_id)
    grads = gradients_p1_tetra(vertices)
    local_rhs = grads @ source.moment
    local_sum = abs(float(local_rhs.sum()))
    local_norm = float(np.linalg.norm(local_rhs))
    if local_sum > tol * max(1.0, local_norm):
        raise ValueError(f"point dipole local RHS is not compatible with Neumann nullspace: sum={local_rhs.sum():g}")
    return resolved_cell_id, cell_dofs, vertices, local_rhs


def get_nonzero_dofs_from_rhs(rhs, tol: float = 1e-14) -> np.ndarray:
    """Return local nonzero dof ids from a DOLFINx Function or PETSc Vec-like RHS."""
    if hasattr(rhs, "x") and hasattr(rhs.x, "array"):
        arr = np.asarray(rhs.x.array, dtype=float)
    elif hasattr(rhs, "getArray"):
        arr = np.asarray(rhs.getArray(readonly=True), dtype=float)
    elif hasattr(rhs, "array"):
        arr = np.asarray(rhs.array, dtype=float)
    else:
        raise TypeError("rhs must be a dolfinx.fem.Function, PETSc Vec, or array-like vector with values")
    return np.flatnonzero(np.abs(arr) > tol).astype(np.int64)


def inspect_point_dipole_location_petsc(
    solver,
    source: PointDipole,
    cell_id: int | None = None,
    tol: float = 1e-10,
    trust_source_cell_id: bool = False,
) -> dict:
    """Return source location and RHS diagnostics in DOLFINx cell ordering."""
    mesh_data = _solver_mesh_data(solver)
    meshdata_located_cell_id = None
    try:
        meshdata_located_cell_id = locate_point_in_mesh(mesh_data, source.position, tol=tol)
    except ValueError:
        pass

    used_cell_id, cell_dofs, dof_coordinates, local_rhs = _point_dipole_local_rhs_in_dof_order(
        solver,
        source,
        cell_id=cell_id,
        trust_source_cell_id=trust_source_cell_id,
        tol=tol,
    )
    if not hasattr(solver, "zero_function"):
        raise TypeError("solver must provide zero_function() to inspect PETSc RHS")
    rhs = solver.zero_function()
    rhs.x.array[:] = 0.0
    rhs.x.array[cell_dofs] += local_rhs
    rhs.x.scatter_forward()
    nonzero_dofs = get_nonzero_dofs_from_rhs(rhs)

    dof_cell_center = dof_coordinates.mean(axis=0)
    meshdata_cell_vertices = None
    meshdata_cell_center = None
    meshdata_dolfinx_center_difference = None
    if 0 <= used_cell_id < mesh_data.num_cells:
        meshdata_cell_vertices = mesh_data.points[mesh_data.cells[used_cell_id]].copy()
        meshdata_cell_center = meshdata_cell_vertices.mean(axis=0)
        meshdata_dolfinx_center_difference = meshdata_cell_center - dof_cell_center

    center_difference_norm = (
        None
        if meshdata_dolfinx_center_difference is None
        else float(np.linalg.norm(meshdata_dolfinx_center_difference))
    )
    ordering_warning = None
    if source.cell_id is not None and int(source.cell_id) != used_cell_id:
        ordering_warning = (
            "source.cell_id is a MeshData cell id and differs from the DOLFINx cell id located from source.position"
        )
    elif center_difference_norm is not None and center_difference_norm > tol:
        ordering_warning = "equal integer MeshData and DOLFINx cell ids have different geometric centers"

    barycentric = barycentric_coordinates_tetra(source.position, dof_coordinates)
    barycentric_min = float(barycentric.min())
    barycentric_sum = float(barycentric.sum())
    return {
        "declared_position": source.position.copy(),
        "declared_cell_id": source.cell_id,
        "meshdata_located_cell_id": meshdata_located_cell_id,
        "used_cell_id": used_cell_id,
        "cell_id": used_cell_id,
        "cell_dofs": cell_dofs.copy(),
        "dof_coordinates": dof_coordinates.copy(),
        "dof_cell_center": dof_cell_center,
        "meshdata_cell_vertices": meshdata_cell_vertices,
        "meshdata_cell_center": meshdata_cell_center,
        "meshdata_dolfinx_center_difference": meshdata_dolfinx_center_difference,
        "meshdata_dolfinx_center_difference_norm": center_difference_norm,
        "ordering_warning": ordering_warning,
        "barycentric_in_dolfinx_cell": barycentric,
        "barycentric_min": barycentric_min,
        "barycentric_sum": barycentric_sum,
        "is_inside_used_dolfinx_cell": bool(
            barycentric_min >= -tol and np.all(barycentric <= 1.0 + tol) and abs(barycentric_sum - 1.0) <= tol
        ),
        "local_rhs": local_rhs.copy(),
        "local_rhs_sum": float(local_rhs.sum()),
        "local_rhs_norm": float(np.linalg.norm(local_rhs)),
        "nonzero_dofs": nonzero_dofs,
        "nonzero_values": np.asarray(rhs.x.array[nonzero_dofs], dtype=float).copy(),
    }


def inspect_point_dipole_rhs_petsc(
    solver,
    source: PointDipole,
    cell_id: int | None = None,
    tol: float = 1e-10,
    trust_source_cell_id: bool = False,
) -> dict:
    """Backward-compatible alias for full point dipole location diagnostics."""
    return inspect_point_dipole_location_petsc(
        solver,
        source,
        cell_id=cell_id,
        tol=tol,
        trust_source_cell_id=trust_source_cell_id,
    )


def compare_meshdata_and_dolfinx_cell_centers(solver, cell_ids=None, max_cells=None) -> dict:
    """Compare centers for equal integer ids in MeshData and local DOLFINx ordering."""
    mesh_data = _solver_mesh_data(solver)
    common_count = min(mesh_data.num_cells, _num_local_dolfinx_cells(solver))
    if cell_ids is None:
        count = common_count if max_cells is None else min(common_count, int(max_cells))
        selected = np.arange(count, dtype=np.int64)
    else:
        selected = np.asarray(cell_ids, dtype=np.int64)
        if selected.ndim != 1:
            raise ValueError("cell_ids must be one-dimensional")
        if selected.size > 0 and (selected.min() < 0 or selected.max() >= common_count):
            raise ValueError("cell_ids contain ids outside the common MeshData/DOLFINx cell range")
        if max_cells is not None:
            selected = selected[: int(max_cells)]

    diffs = np.empty(selected.shape[0], dtype=float)
    for index, selected_cell_id in enumerate(selected):
        selected_cell_id = int(selected_cell_id)
        meshdata_center = mesh_data.points[mesh_data.cells[selected_cell_id]].mean(axis=0)
        _, dolfinx_vertices = _dolfinx_cell_geometry(solver, selected_cell_id)
        diffs[index] = np.linalg.norm(meshdata_center - dolfinx_vertices.mean(axis=0))

    if diffs.size == 0:
        return {
            "max_diff": 0.0,
            "mean_diff": 0.0,
            "num_checked": 0,
            "worst_cell_id": None,
            "cell_ids": selected,
            "diffs": diffs,
        }
    worst_index = int(np.argmax(diffs))
    return {
        "max_diff": float(diffs.max()),
        "mean_diff": float(diffs.mean()),
        "num_checked": int(diffs.size),
        "worst_cell_id": int(selected[worst_index]),
        "cell_ids": selected,
        "diffs": diffs,
    }


def create_cell_marker_function(solver, cell_id: int, value: float = 1.0, name: str = "source_marker"):
    """Create a P1 marker with nonzero values on one local DOLFINx cell's dofs."""
    if not hasattr(solver, "zero_function"):
        raise TypeError("solver must provide zero_function() returning a dolfinx.fem.Function")
    cell_dofs, _ = _dolfinx_cell_geometry(solver, int(cell_id))
    marker = solver.zero_function()
    marker.name = str(name)
    marker.x.array[:] = 0.0
    marker.x.array[cell_dofs] = float(value)
    marker.x.scatter_forward()
    return marker


def assemble_point_dipole_rhs_petsc(
    solver,
    source: PointDipole,
    cell_id: int | None = None,
    trust_source_cell_id: bool = False,
):
    """Assemble a point dipole RHS in the FEniCSx dof ordering expected by ``solver.solve``."""
    _, cell_dofs, _, local_rhs = _point_dipole_local_rhs_in_dof_order(
        solver,
        source,
        cell_id=cell_id,
        trust_source_cell_id=trust_source_cell_id,
    )
    if not hasattr(solver, "zero_function"):
        raise TypeError("solver must provide zero_function() returning a dolfinx.fem.Function")

    rhs = solver.zero_function()
    rhs.x.array[:] = 0.0
    rhs.x.array[cell_dofs] += local_rhs
    rhs.x.scatter_forward()
    return rhs
