# Verification module

Numerical helpers for checking the FEM and forward pipeline:

- `analytic_solutions.py`: homogeneous free-space dipole reference;
- `manufactured.py`: Neumann-compatible cosine solution and forcing;
- `mesh_refinement.py`: structured tetrahedral `MeshData` for the unit cube;
- `convergence.py`: convergence entries, rates, reports, and table formatting.

The module imports without DOLFINx. Real FEM tests are gated by `RUN_DOLFINX_TESTS=1`:

```bash
pytest tests/test_convergence_utils.py
TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp RUN_DOLFINX_TESTS=1 \
  pytest tests/test_forward_convergence.py tests/test_poisson_manufactured_solution.py
```

Additional integration tests cover the cached P1 locator, source localization, FEM/Green reciprocity, single-dipole inverse recovery, and benchmark handoff.
