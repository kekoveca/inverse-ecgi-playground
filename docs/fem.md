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
- кэшированный scalar-P1 node↔DOF mapping;
- кэшированный local-cell tetra locator для source/candidate lookup.

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

### Node-to-DOF mapping

Для serial scalar P1 доступен coordinate-verified mapping:

```python
mapping = solver.p1_node_dof_mapping()
node_to_dof = mapping.node_to_dof
dof_to_node = mapping.dof_to_node
```

`build_node_to_dof_map_p1(solver)` делегирует этому кэшу. Mapping используется `forward` и `green`, когда numpy operator в MeshData node ordering нужно связать с DOLFINx Function. В distributed MPI layout helper намеренно выбрасывает ошибку вместо предположения об ownership/ghost ids.

### P1 tetra locator

```python
locator = solver.p1_tetra_locator()
cell_ids, barycentric = locator.locate_points(points, return_barycentric=True)
cell_dofs, vertices = locator.cell_geometry(cell_ids)
grads_phi = locator.basis_gradients(cell_ids)
```

`DOLFINxP1TetraLocator` один раз кэширует dof coordinates, local cell dofs, vertices, centers и KD-tree. KD-tree лишь упорядочивает candidate cells; итоговое попадание всегда подтверждается barycentric containment test. Возвращаемые ids относятся к owned local DOLFINx cells. Поддерживается только scalar P1 tetra space.

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

`destroy()` также сбрасывает FEM mapping/locator caches. Не переиспользуйте полученные locator/mapping objects после уничтожения solver.

В MPI-программе создание, solve и destroy должны выполняться согласованно всеми ranks communicator.

## Manufactured solution convergence

Гладкая проверка solver использует unit cube и

```text
u_exact = cos(2 pi x) cos(2 pi y) cos(2 pi z)
-Delta u_exact = 12 pi^2 u_exact
```

Нормальная производная равна нулю на всех гранях unit cube, поэтому функция совместима с homogeneous Neumann boundary condition.

`test_poisson_manufactured_solution.py` решает задачу на refinement levels `n=4, 8, 16`, выравнивает constant gauge и проверяет:

- монотонное уменьшение L2 error;
- минимальный наблюдаемый rate больше `1.0`.

Для P1 на smooth solution ожидается близкий к quadratic L2 rate, но test использует мягкий threshold, чтобы не зависеть от coarse-grid и solver details.
