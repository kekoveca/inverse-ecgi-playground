# Green functions

## Purpose

`green` converts electrode measurement functionals into reusable transfer matrices for point-dipole sources. It is the mathematical layer between the forward solver and a future inverse solver.

```text
MeasurementOperator M
        |
        v
K G_i = M_i^T
        |
        v
grad G_i(x_j)
        |
        v
A[j, i, :] and g = A_j p
```

## Mathematical convention

The project uses

```text
K u = b_source
b_source,a = grad(phi_a)(x0) dot p
```

and the discrete reciprocal Green problems

```text
K G_i = M_i^T
A[j, i, :] = grad G_i(x_j)
g_i = p dot grad G_i(x_j).
```

The initial transfer sign is `+1`. `compare_forward_and_green` compares both `A_j @ p` and `-A_j @ p`, reports their relative errors and selects `best_sign`. The sign in `sources` is not modified implicitly.

## Measurement RHS compatibility

A pure Neumann solve requires every RHS to be orthogonal to constants:

```text
sum(M_i) = 0.
```

Average reference satisfies this condition. `measurement_matrix_row_sums` and `check_measurement_matrix_compatibility` expose the check, and `GreenSolver.solve_all()` rejects an incompatible matrix before solving.

## Ordering warning

`MeasurementOperator.matrix()` is assembled in **MeshData node ordering**. DOLFINx functions and PETSc vectors use **DOLFINx DOF ordering**. These permutations are not assumed equal.

`create_green_rhs_function` therefore:

1. extracts one row of `M` in MeshData node ordering;
2. builds a coordinate-verified scalar-P1 `node_id -> dof_id` map;
3. writes each nodal coefficient to the corresponding DOLFINx dof.

Do not copy a row of `M` directly into `Function.x.array`. The current coordinate mapping is a serial scalar-P1 tetra MVP and raises a clear error for unsupported distributed layouts.

## Solving a Green basis

```python
from green import GreenSolver, build_green_transfer_matrix

measurement_operator = forward.measurement_operator

green_solver = GreenSolver(
    poisson_solver=solver,
    measurement_operator=measurement_operator,
)
green_basis = green_solver.solve_all()

transfer = build_green_transfer_matrix(
    poisson_solver=solver,
    green_basis=green_basis,
    candidate_points=geometry.source_region.candidate_points,
)

A0 = transfer.matrix_for_candidate(0)
g_pred = transfer.predict(0, moment=[0.0, 0.0, 1.0])
```

`GreenSolver` reuses the stiffness matrix and KSP owned by `NeumannPoissonSolver`. `GreenBasis.functions` contains one DOLFINx function per solved measurement row when `keep_functions=True`.

Candidate points are located again in DOLFINx cell ordering. `SourceRegion.candidate_cell_ids` are MeshData cell ids and must not be passed as DOLFINx ids without an explicit verified mapping.

## Consistency check

```python
from green import compare_forward_and_green

diagnostics = compare_forward_and_green(
    result_forward,
    transfer,
    candidate_index=0,
    moment=result_forward.source.moment,
)

print(diagnostics["best_sign"])
print(diagnostics["best_rel_error"])
```

The small-mesh integration test requires `rel_error_plus < 1e-6` and currently confirms `best_sign == +1`, so the implemented discrete convention is `g = A_j @ p`. The diagnostic still evaluates both signs to make convention regressions visible.

## Cache

```python
from green import save_green_transfer_matrix, load_green_transfer_matrix

save_green_transfer_matrix(transfer, "output/green_transfer.npz")
transfer = load_green_transfer_matrix("output/green_transfer.npz")
```

The cache stores `A`, candidate points, DOLFINx candidate cell ids, sign and JSON metadata. Green functions themselves are not stored.

## Inverse handoff

`GreenTransferMatrix` is the direct input to the `inverse` module:

```python
from inverse import SingleDipoleInverseSolver

inverse_solver = SingleDipoleInverseSolver(transfer, lambda_reg=1e-10)
inverse_result = inverse_solver.solve(observed_measurements)
```

Inverse reconstruction uses `transfer.matrix_for_candidate(j)`, so `GreenTransferMatrix.sign` remains the single place where the Green/FEM sign convention is represented.
