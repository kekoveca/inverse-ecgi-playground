# Forward module

Full guide: [../docs/forward.md](../docs/forward.md).

`forward` composes the point-dipole RHS, Neumann Poisson solver, measurement operator, result object, and ParaView export:

```text
PointDipole -> RHS -> FEM solve -> nodal potential -> measurements -> ForwardResult
```

## Example

```python
from fem import NeumannPoissonSolver
from forward import ForwardSolver, export_forward_result_to_vtx
from sources import PointDipole

solver = NeumannPoissonSolver(volume_mesh, degree=1, sigma=1.0)
try:
    pipeline = ForwardSolver(solver, electrodes=electrodes, reference="average")
    source = PointDipole(position=[0.0, 0.0, 0.0], moment=[0.0, 0.0, 1.0])
    result = pipeline.solve(source)
    export_forward_result_to_vtx(result, "output/potential.bp")
finally:
    solver.destroy()
```

The stiffness matrix is assembled by `fem` and reused. DOLFINx DOF values are mapped into MeshData node ordering before `MeasurementOperator` evaluation.

## ParaView export

VTX/BP is the preferred format. XDMF may create both `.xdmf` and `.h5`; open the `.xdmf` file and keep both files together.

```python
from forward import export_electrode_markers_to_vtx, export_forward_result_to_xdmf

export_forward_result_to_xdmf(result, "output/potential.xdmf")
export_electrode_markers_to_vtx(solver, electrodes, "output/electrodes.bp")
```

`electrodes.bp` marks the nearest FEM DOF to each electrode. It is a diagnostic nodal field, not an exact point cloud.

The pure-Neumann potential is defined only up to a constant. The FEM solver fixes a gauge, and average reference removes the constant from electrode measurements. Current FEM/Green tests confirm transfer sign `+1`.
