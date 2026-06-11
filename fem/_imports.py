from __future__ import annotations


def require_fenicsx():
    """Import FEniCSx stack lazily and raise a clear error if it is missing."""
    try:
        from mpi4py import MPI
        from petsc4py import PETSc
        import basix.ufl
        import dolfinx
        from dolfinx import fem, mesh as dmesh
        from dolfinx.fem import petsc as fem_petsc
        import ufl
    except ImportError as exc:  # pragma: no cover - depends on external stack
        raise ImportError(
            "The fem module requires FEniCSx: dolfinx, basix, ufl, mpi4py and petsc4py. "
            "Install/use a FEniCSx environment before running the solver."
        ) from exc

    return {
        "MPI": MPI,
        "PETSc": PETSc,
        "basix_ufl": basix.ufl,
        "dolfinx": dolfinx,
        "fem": fem,
        "dmesh": dmesh,
        "fem_petsc": fem_petsc,
        "ufl": ufl,
    }
