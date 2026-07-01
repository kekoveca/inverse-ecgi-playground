# API overview

This page lists the public imports exported by each package `__init__.py`. Detailed behavior and limitations live in the module guides.

## Typical pipeline imports

```python
from geometry import ElectrodeSet, MeshData, SourceRegion, TorsoGeometry, read_gmsh_meshio
from fem import NeumannPoissonSolver
from sources import PointDipole, assemble_point_dipole_rhs_petsc
from measurements import MeasurementOperator, build_measurement_operator
from forward import ForwardSolver, export_forward_result_to_vtx
from green import GreenSolver, build_green_transfer_matrix
from inverse import SingleDipoleInverseSolver
from benchmark import run_forward_benchmark, run_inverse_benchmark
```

## geometry

Data and validation:

- `MeshData`, `MeshQualityReport`, `quality_report`, `tetra_volumes`;
- `ElectrodeSet`, `ElectrodePlacementReport`, `electrode_placement_report`;
- `SourceRegion`, `TorsoGeometry`, `GeometryValidationReport`, `validate_torso_geometry`;
- `read_gmsh_meshio`, `load_npz_mesh`, `save_npz_mesh`;
- `AffineTransform`, `transform_torso_geometry`;
- `plot_mesh`, `plot_source_region`, `plot_torso_geometry`.

`TaggedMesh` is not a public class. Gmsh tags and multi-block behavior are implemented by `MeshData`.

## fem

- `NeumannPoissonSolver` / `FEMProblem`;
- `FunctionSpaceFactory`, `StiffnessOperator`, `LinearSolver`, `SolverDiagnostics`;
- `NeumannNullspaceHandler`, `ConstantNullspace`;
- `create_dolfinx_mesh`, `infer_cell_type`;
- `DOLFINxP1Mapping`, `build_p1_node_dof_mapping`, `build_node_to_dof_map_p1`;
- `DOLFINxP1TetraLocator`, `get_p1_tetra_locator`.

## sources

- `PointDipole`;
- `gradients_p1_tetra`, `barycentric_coordinates_tetra`, `barycentric_boundary_flags`;
- `point_in_tetra`, `tetra_volume`, `tetra_signed_volume`;
- `assemble_point_dipole_rhs_numpy`, `assemble_point_dipole_rhs_petsc`;
- `locate_point_in_mesh`, `locate_points_in_mesh`, `locate_point_in_dolfinx_p1_tetra_mesh`;
- `inspect_point_dipole_location_petsc`, `inspect_point_dipole_rhs_petsc`;
- `compare_meshdata_and_dolfinx_cell_centers`, `create_cell_marker_function`;
- `check_rhs_compatibility`, `rhs_compatibility_error`, `get_nonzero_dofs_from_rhs`.

## measurements

- `MeasurementOperator`, `build_measurement_operator`;
- `build_point_interpolation_matrix`, `evaluate_at_points`;
- `locate_points_in_tetra_mesh`, `locate_electrodes_in_mesh`;
- `measure_nodal_values`, `measure_raw_nodal_values`;
- `reference_matrix`, `average_reference_matrix`, `apply_reference`, `apply_average_reference`;
- `ElectrodeProjectionReport`, `TetraVolumeLocator`, `CentralSurfaceProjector`;
- `central_project_electrodes_to_surface`, `central_project_point_to_surface`;
- `boundary_triangle_mesh_from_tetra_mesh`.

## forward

- `ForwardSolver`, `ForwardResult`, `extract_nodal_values`;
- `export_dolfinx_function_to_vtx`;
- `export_potential_to_vtx`, `export_forward_result_to_vtx`;
- `export_potential_to_xdmf`, `export_forward_result_to_xdmf`;
- `create_electrode_marker_function`, `inspect_electrode_marker_mapping`;
- `export_electrode_markers_to_vtx`.

## green

- `GreenSolver`, `GreenBasis`, `GreenSolveInfo`;
- `GreenTransferMatrix`, `build_green_transfer_matrix`;
- `create_green_rhs_function`, `create_function_from_meshdata_nodal_values`;
- `get_measurement_matrix`, `extract_measurement_rhs_row`;
- `measurement_matrix_row_sums`, `check_measurement_matrix_compatibility`;
- `gradient_on_dolfinx_cell`, `gradients_at_candidate_cells`, `gradients_at_candidate_points`;
- `locate_candidate_points_in_dolfinx`;
- `compare_forward_and_green`, `infer_green_sign_from_cases`;
- `save_green_transfer_matrix`, `load_green_transfer_matrix`.

`green.build_node_to_dof_map_p1` remains a compatibility export and delegates to `fem`.

## inverse

- `SingleDipoleInverseSolver`, `solve_single_dipole_inverse`;
- `CandidateInverseSolution`, `SingleDipoleInverseResult`;
- `solve_tikhonov_moment`, `residual_vector`, `residual_norm`, `relative_residual`, `condition_number`;
- `localization_error`, `moment_relative_error`, `moment_angle_error_deg`;
- `inverse_reconstruction_metrics`, `is_successful_localization`;
- `summarize_inverse_result`, `format_inverse_summary`.

## benchmark

Forward benchmark:

- `ForwardBenchmarkScenario`, `ForwardBenchmarkRunner`, `run_forward_benchmark`;
- `ForwardBenchmarkRecord`, `ForwardBenchmarkResult`, `save_forward_benchmark_result`;
- `SourceSet`, source generators and `axis_moments`;
- `ElectrodeSubset` and electrode subset selectors;
- `NoiseModel`, `NoNoise`, `AbsoluteGaussianNoise`, `RelativeGaussianNoise`;
- forward/noise metric functions.

Inverse benchmark:

- `InverseBenchmarkScenario`, `InverseBenchmarkRunner`, `run_inverse_benchmark`;
- `InverseBenchmarkRecord`, `InverseBenchmarkResult`, `save_inverse_benchmark_result`;
- `filter_forward_result_by_electrode_set`.

## verification

- `create_unit_cube_meshdata`;
- `u_exact_neumann_cosine`, `rhs_neumann_cosine`;
- `homogeneous_free_space_dipole_potential`;
- `ConvergenceEntry`, `ConvergenceReport`, `estimate_rates`, `build_convergence_report`, `format_convergence_report`.

## performance

- `PerformanceTimer`, `TimingRecord`;
- `get_process_memory_mb`, `estimate_array_memory_mb`;
- `profile_callable`, `run_cprofile`;
- `format_timing_table`, `save_timing_csv`, `save_timing_json`.
