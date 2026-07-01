# Debugging

## ParaView shows weird artifacts

Проверяйте pipeline по слоям:

1. число и значения ненулевых RHS dofs;
2. положение источника в DOLFINx cell ordering;
3. соответствие MeshData/DOLFINx cell ids;
4. локальный DOF ordering;
5. source marker в ParaView;
6. положения электродов и barycentric interpolation;
7. PETSc convergence/nullspace diagnostics.

Не меняйте знак RHS, пока не исключены ошибки ordering и location.

## Check RHS localization

```python
from sources import inspect_point_dipole_rhs_petsc

info = inspect_point_dipole_rhs_petsc(solver, source)
print(info["nonzero_dofs"])
print(info["nonzero_values"])
print(info["local_rhs"])
print(info["local_rhs_sum"])
```

Для scalar P1 tetra local RHS записывается в четыре cell dofs. Для общего направления момента ожидаются четыре ненулевых значения; специальное направление может дать точный ноль на отдельном dof. Сумма local RHS должна быть близка к нулю.

## Check source location

```python
from sources import inspect_point_dipole_location_petsc

info = inspect_point_dipole_location_petsc(solver, source)

assert info["is_inside_used_dolfinx_cell"]
assert abs(info["barycentric_sum"] - 1.0) < 1e-8
assert info["barycentric_min"] > -1e-8

print(info["declared_position"])
print(info["meshdata_located_cell_id"])
print(info["used_cell_id"])
print(info["dof_cell_center"])
print(info["ordering_warning"])
```

`source.cell_id` относится к MeshData ordering и по умолчанию не используется PETSc assembler.

Для нескольких точек можно проверить cached locator напрямую:

```python
locator = solver.p1_tetra_locator()
cell_ids, barycentric = locator.locate_points(points, return_barycentric=True)
print(locator.metadata)
```

Возвращаемые ids локальны для DOLFINx solver, а не для `MeshData`.

## Export source marker

```python
from forward import export_dolfinx_function_to_vtx
from sources import create_cell_marker_function

marker = create_cell_marker_function(solver, info["used_cell_id"])
export_dolfinx_function_to_vtx(
    marker,
    "output/source_marker.bp",
    name="source_marker",
)
```

Откройте `output/source_marker.bp` в ParaView. Marker использует P1 nodal values на dofs выбранной ячейки, поэтому его support может быть виден и на соседних cells, разделяющих эти вершины.

## Check mesh/cell ordering

```python
from sources import compare_meshdata_and_dolfinx_cell_centers

report = compare_meshdata_and_dolfinx_cell_centers(solver, max_cells=1000)
print(report["max_diff"])
print(report["mean_diff"])
print(report["worst_cell_id"])
```

Если differences существенно больше floating-point tolerance, одинаковые integer ids относятся к разным ячейкам. Не используйте MeshData cell ids как DOLFINx ids.

На текущей `torso.msh` такое различие было подтверждено диагностикой: source cell id в MeshData и DOLFINx различаются.

## Export RHS and potential

```python
from forward import export_dolfinx_function_to_vtx, export_forward_result_to_vtx
from sources import assemble_point_dipole_rhs_petsc

rhs = assemble_point_dipole_rhs_petsc(solver, source)
export_dolfinx_function_to_vtx(rhs, "output/rhs.bp", name="rhs")

result = forward.solve(source)
export_forward_result_to_vtx(result, "output/potential.bp")
```

Сначала проверьте `source_marker.bp`, затем `rhs.bp`, и только потом интерпретируйте `potential.bp`.

## Check electrodes

```python
from geometry import electrode_placement_report, validate_torso_geometry

geometry_report = validate_torso_geometry(torso_geometry)
placement = electrode_placement_report(torso_geometry.electrodes, torso_geometry.volume_mesh)

print(geometry_report.is_valid)
print(geometry_report.errors)
print(placement.mean_distance_to_nearest_node)
print(placement.max_distance_to_nearest_node)
```

Также проверьте `MeasurementOperator.electrode_cell_ids` и `electrode_barycentric`. Электроды должны лежать внутри выбранной volume mesh либо быть предварительно спроецированы в допустимую область.

Если электроды находятся вне volume mesh, `build_measurement_operator` по умолчанию делает центральную проекцию на `surface_mesh` или на boundary, извлеченный из tetra mesh:

```python
op = build_measurement_operator(
    torso_geometry.volume_mesh,
    torso_geometry.electrodes,
    surface_mesh=torso_geometry.surface_mesh,
)

print(op.metadata["electrode_projection"])
```

В projection report значение `surface_cell_ids[i] == -1` нормально для электрода, который не проецировался. Смотрите одновременно `projected_mask`, projection distance и отдельную nearest-surface диагностику tutorial example.

`surface_mesh.num_points` тоже может выглядеть неожиданно большим: extracted triangle mesh иногда сохраняет полный global point array. Фактическое число используемых surface vertices:

```python
num_surface_used_vertices = np.unique(surface_mesh.cells.ravel()).size
```

## Checking electrode placement in ParaView

The full inverse example exports `electrodes.bp`, a diagnostic scalar field on
the FEM mesh. It marks the nearest DOLFINx dof to each electrode position:

```python
from forward import export_electrode_markers_to_vtx, inspect_electrode_marker_mapping

info = inspect_electrode_marker_mapping(solver, electrodes)
print(info["num_unique_dofs"])
print(info["num_collisions"])
print(info["max_distance"])

export_electrode_markers_to_vtx(
    solver,
    electrodes,
    "output/electrodes.bp",
    value_mode="index",
)
```

Open `potential.bp`, `source_marker.bp` and `electrodes.bp` together in
ParaView. Marker values `1, 2, 3, ...` correspond to electrode index + 1. This
is not an exact point cloud: if an electrode lies between mesh nodes, the marker
appears at the nearest dof. For exact coordinates and distances, inspect
`electrode_marker_mapping.csv`.

## Solver diagnostics

```python
print(solver.diagnostics.converged_reason)
print(solver.diagnostics.residual_norm)
print(solver.diagnostics.nullspace_test_passed)
```

Положительный `converged_reason` и успешный nullspace test нужны до анализа физической формы поля.

## Green RHS is incompatible

For pure Neumann Green solves every measurement row must sum to zero:

```python
from green import measurement_matrix_row_sums

row_sums = measurement_matrix_row_sums(forward.measurement_operator)
print(np.max(np.abs(row_sums)))
```

`reference="none"` обычно несовместим; используйте `average` или корректно настроенный `single` reference. Не исправляйте это произвольным вычитанием после Green solve: compatibility должна быть свойством measurement functional.

## Inverse result has wrong sign or location

Сначала сравните ordinary forward и Green prediction в true candidate:

```python
diagnostics = compare_forward_and_green(
    forward_result,
    transfer,
    candidate_index=true_index,
    moment=source.moment,
)
print(diagnostics)
```

Если `best_rel_error` мал, но inverse выбирает другую позицию, проверьте ambiguity/top candidates, condition numbers и noise. Если error велик, проверьте transfer provenance: geometry, electrode order, reference, `measurement_row_indices`, `sigma`, candidates и `sign`. Inverse использует `transfer.matrix_for_candidate()` и не меняет sign сам.

## If forward solution looks unstable

Запустите проверки от дешёвых к дорогим:

```bash
pytest test_convergence_utils.py test_measurements_module.py

TMPDIR=/tmp OMPI_MCA_orte_tmpdir_base=/tmp RUN_DOLFINX_TESTS=1 \
  pytest test_forward_convergence.py test_poisson_manufactured_solution.py
```

Интерпретация:

- manufactured L2 convergence не проходит — проблема в mesh/FEM assembly/nullspace/solver;
- manufactured test проходит, но linearity или scaling не проходят — проверяйте RHS assembly и KSP tolerance;
- linearity проходит, но refinement measurements нестабилен — проверяйте source location относительно mesh skeleton и observation ordering;
- все verification tests проходят, но torso field выглядит странно — проверяйте physical geometry, conductivity model, source marker и electrodes.

Не используйте source point, лежащий точно на refinement grid vertex, для cell-local point-dipole convergence: такая точка принадлежит нескольким тетраэдрам.

## If transfer matrix construction is slow

Отделите point location от gradient evaluation:

```bash
python3 scripts/profile_components.py \
  --component green-transfer \
  --mesh torso_refined.msh \
  --num-candidates 50 \
  --num-measurements 16 \
  --output output/green_transfer_profile
```

Profile сравнивает build с lookup и с заранее найденными `candidate_cell_ids`. В текущей реализации оба пути используют cached `DOLFINxP1TetraLocator`, а basis gradients вычисляются batch-операцией. Если stage всё ещё дорогой, отдельно проверьте первый build locator, число candidates/functions и память retained Green functions.
