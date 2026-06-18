from __future__ import annotations

from pathlib import Path

from .result import ForwardResult


def _require_dolfinx_io():
    try:
        from dolfinx import io
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("exporting FEM potentials requires dolfinx") from exc
    return io


def _function_mesh(potential):
    function_space = getattr(potential, "function_space", None)
    mesh = getattr(function_space, "mesh", None)
    vector = getattr(potential, "x", None)
    if mesh is None:
        raise TypeError("potential must be a dolfinx.fem.Function with function_space.mesh")
    if vector is None or not hasattr(vector, "array"):
        raise TypeError("potential must be a dolfinx.fem.Function with x.array values")
    return mesh


def export_potential_to_xdmf(potential, path, name: str = "potential", time: float = 0.0) -> Path:
    """Export a DOLFINx Function potential to XDMF for ParaView."""
    mesh = _function_mesh(potential)
    io = _require_dolfinx_io()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        potential.name = str(name)
    except AttributeError as exc:
        raise TypeError("potential must be a writable dolfinx.fem.Function") from exc

    with io.XDMFFile(mesh.comm, str(output_path), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(potential, float(time))
    return output_path


def export_dolfinx_function_to_vtx(
    function,
    path,
    name: str = "field",
    time: float = 0.0,
    engine: str = "BP4",
) -> Path:
    """Export a DOLFINx Function to VTX/BP for ParaView."""
    mesh = _function_mesh(function)
    io = _require_dolfinx_io()
    if not hasattr(io, "VTXWriter"):
        raise ImportError("dolfinx.io.VTXWriter is not available; install DOLFINx with ADIOS2 support")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        function.name = str(name)
    except AttributeError as exc:
        raise TypeError("function must be a writable dolfinx.fem.Function") from exc

    with io.VTXWriter(mesh.comm, str(output_path), [function], engine=engine) as vtx:
        vtx.write(float(time))
    return output_path


def export_potential_to_vtx(
    potential,
    path,
    name: str = "potential",
    time: float = 0.0,
    engine: str = "BP4",
) -> Path:
    """Export a DOLFINx Function potential to VTX/BP for ParaView."""
    return export_dolfinx_function_to_vtx(potential, path, name=name, time=time, engine=engine)


def export_forward_result_to_xdmf(
    result: ForwardResult,
    path,
    name: str = "potential",
    time: float = 0.0,
) -> Path:
    """Write result potential plus mesh to XDMF/HDF5 and return the XDMF path.

    Open the ``.xdmf`` file in ParaView and keep its companion ``.h5`` nearby.
    """
    return export_potential_to_xdmf(result.potential, path, name=name, time=time)


def export_forward_result_to_vtx(
    result: ForwardResult,
    path,
    name: str = "potential",
    time: float = 0.0,
    engine: str = "BP4",
) -> Path:
    """Write result potential to VTX/BP and return the output path.

    VTX/BP is the preferred ParaView fallback when XDMF is unstable.
    """
    return export_potential_to_vtx(result.potential, path, name=name, time=time, engine=engine)
