# FEM

`fem` — DOLFINx/PETSc-backed ядро для scalar Poisson problem с чистыми условиями Неймана.

## NeumannPoissonSolver

`NeumannPoissonSolver` является публичным именем `FEMProblem`. Он собирает систему для

```text
-div(sigma grad(u)) = f
```

с постоянной scalar conductivity `sigma` и однородным Neumann boundary condition. В MVP поддерживается только P1 Lagrange (`degree=1`).

Solver создаёт и хранит:

- DOLFINx mesh (`domain`);
- function space (`V`);
- stiffness matrix (`K`, alias `A`);
- PETSc constant nullspace;
- KSP solver, переиспользуемый для нескольких RHS.

## Pure Neumann nullspace

Для чистой задачи Неймана

```text
K 1 = 0
```

поэтому решение определено с точностью до константы. Это математическое свойство задачи, а не ошибка сборки.

`NeumannNullspaceHandler`:

1. создаёт PETSc `NullSpace(constant=True)`;
2. прикрепляет его к `K`;
3. удаляет constant component из RHS;
4. проверяет совместимость RHS;
5. фиксирует gauge решения вычитанием среднего.

Для совместимого RHS ожидается нулевая сумма в дискретном constant mode. `solve` по умолчанию выполняет проекцию и проверку.

## Mesh conversion

```text
geometry.MeshData -> dolfinx.mesh.Mesh -> P1 FunctionSpace
```

Преобразование выполняет `create_dolfinx_mesh`.

### Critical ordering warning

После преобразования не гарантируется:

```text
MeshData node_id == DOLFINx dof_id
MeshData cell_id == DOLFINx cell_id
```

Не записывайте значения, индексированные по `MeshData`, напрямую в PETSc/FEniCSx vectors. Для cell-local операций используйте `V.dofmap.cell_dofs(dolfinx_cell_id)` и координаты `V.tabulate_dof_coordinates()`.

## Solver usage

```python
from fem import NeumannPoissonSolver

solver = NeumannPoissonSolver(
    mesh=volume_mesh,
    degree=1,
    sigma=1.0,
    ksp_type="cg",
    pc_type="hypre",
)

try:
    rhs = solver.zero_function()
    potential = solver.solve(rhs)
finally:
    solver.destroy()
```

`rhs` может быть `dolfinx.fem.Function`, PETSc `Vec` или поддерживаемый vector-like object. В примере решается тривиальная задача с нулевым RHS; для point dipole используйте `sources.assemble_point_dipole_rhs_petsc`.

Матрица `K` и KSP setup не пересобираются между вызовами `solve`.

## Diagnostics

После сборки и решения доступен `solver.diagnostics`:

```python
print(solver.diagnostics.ksp_type)
print(solver.diagnostics.pc_type)
print(solver.diagnostics.converged_reason)
print(solver.diagnostics.residual_norm)
print(solver.diagnostics.nullspace_test_passed)
```

Положительный `converged_reason` означает успешную сходимость PETSc KSP. `nullspace_test_passed` показывает результат проверки constant nullspace при создании solver.

## Resource lifetime

PETSc objects освобождаются явно:

```python
solver.destroy()
```

В MPI-программе создание, solve и destroy должны выполняться согласованно всеми ranks communicator.
