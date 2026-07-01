# verification

Численные helpers для проверки FEM и forward pipeline.

- `analytic_solutions.py` — homogeneous free-space dipole reference.
- `manufactured.py` — Neumann-compatible cosine solution и forcing.
- `mesh_refinement.py` — structured tetrahedral `MeshData` для unit cube.
- `convergence.py` — convergence entries, rates, reports и table formatting.

Модуль импортируется без DOLFINx. Реальные FEM tests находятся в `test_forward_convergence.py` и `test_poisson_manufactured_solution.py` и запускаются с `RUN_DOLFINX_TESTS=1`.

Дополнительные integration tests проверяют cached P1 locator, source localization, FEM/Green reciprocity, single-dipole inverse и benchmark handoff. Полный DOLFINx прогон:

```bash
TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp RUN_DOLFINX_TESTS=1 pytest
```
