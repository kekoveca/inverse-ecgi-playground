# Inverse

## Purpose

`inverse` reconstructs one point dipole from observed electrode measurements using an already built `GreenTransferMatrix`.

It does not build Green functions, solve the FEM problem, or optimize source position continuously. It searches over the discrete candidate points stored in the transfer matrix.

## Mathematical model

For candidate source point `x_j`:

```text
g ≈ A_j p
```

where:

- `A_j = transfer.matrix_for_candidate(j)` has shape `(num_measurements, 3)`;
- `p` is the unknown dipole moment;
- `g` is the observed referenced measurement vector.

For each candidate, the solver computes:

```text
p_j = argmin ||A_j p - g||² + λ ||p||²
```

With `λ = 0`, `np.linalg.lstsq` is used. With `λ > 0`, the normal equations are regularized:

```text
(A_j.T A_j + λ I) p_j = A_j.T g
```

The best candidate is the one with minimal residual norm.

## API example

```python
from inverse import SingleDipoleInverseSolver

inverse_solver = SingleDipoleInverseSolver(
    transfer_matrix=transfer,
    lambda_reg=1e-10,
    reference="average",
)

result = inverse_solver.solve(result_forward.measurements)

print(result.estimated_position)
print(result.estimated_cell_id)
print(result.estimated_moment)
print(result.relative_residual)
```

Convenience wrapper:

```python
from inverse import solve_single_dipole_inverse

result = solve_single_dipole_inverse(
    transfer,
    measurements,
    lambda_reg=1e-10,
    reference="average",
)
```

## Results

`CandidateInverseSolution` stores one candidate LS solution:

- candidate index;
- candidate position;
- DOLFINx candidate cell id;
- estimated moment;
- residual norm;
- relative residual;
- condition number.

`SingleDipoleInverseResult` stores the observed measurements, all solved candidate solutions and the best solution. `to_summary_dict()` intentionally omits full residual and moment maps.

Use:

```python
residuals = result.residual_map()
moments = result.moment_map()
```

when full maps are needed.

## Metrics

Available reconstruction metrics:

- `localization_error`;
- `moment_relative_error`;
- `moment_angle_error_deg`;
- `inverse_reconstruction_metrics`;
- `is_successful_localization`.

Example:

```python
from inverse import inverse_reconstruction_metrics

metrics = inverse_reconstruction_metrics(
    result,
    true_position=source.position,
    true_moment=source.moment,
    localization_threshold=1e-3,
)
```

## Sign convention

The inverse solver always uses:

```python
transfer.matrix_for_candidate(j)
```

It does not inspect or correct raw `transfer.A[j]`. Any FEM/Green sign convention must be represented by `GreenTransferMatrix.sign`, not by ad hoc sign changes in inverse.

Current Green consistency tests confirm the project convention:

```text
g = A_j @ p
```

## Benchmark integration

The inverse module solves one measurement vector. The benchmark inverse runner applies it to many `ForwardBenchmarkRecord` objects:

```text
synthetic forward measurements
  + Green transfer matrix
  -> inverse reconstruction
  -> localization/moment metrics
```

```python
from benchmark import run_inverse_benchmark, save_inverse_benchmark_result

inverse_result = run_inverse_benchmark(
    forward_result,
    transfer,
    lambda_reg=1e-10,
    localization_threshold=20.0,
)

save_inverse_benchmark_result(
    inverse_result,
    "results/inverse",
)
```

`benchmark` records aggregate localization, moment and residual metrics. No multi-dipole inverse or continuous nonlinear optimization is implemented yet.
