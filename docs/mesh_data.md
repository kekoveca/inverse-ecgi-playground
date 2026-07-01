# MeshData: Gmsh/meshio сетки с physical groups

`geometry.mesh_model.MeshData` хранит сетки, прочитанные из Gmsh через `meshio`, вместе с physical groups. Этот слой нужен, когда в одном файле лежат разные блоки ячеек: например `tetra` для объема торса и `triangle` для поверхности.

Для downstream-кода, которому нужен один конкретный блок, multi-block `MeshData` преобразуется в single-block `MeshData` через `to_mesh_data(...)`.

## Основные сущности

- `MeshData`: единый контейнер для координат, cell blocks, physical tags и metadata.
- `read_gmsh_meshio(path, dim)`: чтение `.msh` через `meshio`.
- `_field_data_to_tuples(field_data)`: внутренняя конвертация `meshio.field_data`.

## Внутренняя конвенция `field_data`

Внутри проекта `field_data` всегда хранится в формате Gmsh API:

```python
field_data: dict[str, tuple[int, int]]
# name -> (dim, tag)
```

Примеры:

```python
field_data["domain"] == (3, 1)    # 3D volume group, tag=1
field_data["boundary"] == (2, 2)  # 2D surface group, tag=2
```

Эта конвенция важна: первый элемент пары это геометрическая размерность physical group, второй элемент это physical tag.

## Отличие от `meshio`

`meshio` обычно возвращает `field_data` в другом порядке:

```text
# meshio convention
name -> (tag, dim)
```

Поэтому `read_gmsh_meshio()` вызывает `_field_data_to_tuples()` и переворачивает пары:

```text
meshio:   (tag, dim)
internal: (dim, tag)
```

Например:

```python
from geometry.mesh_model import _field_data_to_tuples
import numpy as np

converted = _field_data_to_tuples(
    {
        "domain": np.array([1, 3]),
        "boundary": np.array([2, 2]),
    }
)

assert converted["domain"] == (3, 1)
assert converted["boundary"] == (2, 2)
```

Если вы создаете `MeshData` вручную, передавайте уже внутренний порядок `(dim, tag)`.

## Структура multi-block `MeshData`

```python
from geometry import MeshData
import numpy as np

mesh = MeshData.from_cell_blocks(
    points=np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    cell_blocks={
        "tetra": np.array([[0, 1, 2, 3]]),
    },
    cell_tags={
        "tetra": np.array([1]),
    },
    field_data={
        "domain": (3, 1),
    },
)
```

Поля:

- `dim`: рабочая размерность координат, обычно `2` или `3`.
- `points`: массив узлов формы `(n_nodes, dim)`.
- `cells`: connectivity активного блока.
- `cell_type`: тип активного блока.
- `cell_blocks`: словарь `cell_type -> connectivity`, например `"tetra"`, `"triangle"`, `"line"`.
- `cell_tags`: словарь `cell_type -> physical tag per cell`.
- `field_data`: словарь `physical_name -> (dim, tag)`.
- `metadata`: произвольные метаданные.

Если для блока ячеек нет `cell_tags`, контейнер создаст нулевые tags для всех ячеек этого блока.

Для совместимости доступны aliases:

- `mesh.coords == mesh.points`;
- `mesh.dim == mesh.geometric_dim`.

## Physical group API

```python
assert mesh.physical_dimension("domain") == 3
assert mesh.physical_tag("domain") == 1
```

`physical_dimension(name)` возвращает первый элемент `field_data[name]`.

`physical_tag(name)` возвращает второй элемент `field_data[name]`.

Если physical group неизвестна, методы бросают `KeyError` со списком доступных групп.

## Фильтрация cell blocks

`cell_block(cell_type, physical_name=None)` возвращает connectivity-массив для указанного типа ячеек. Если передан `physical_name`, ячейки фильтруются по physical tag:

```python
domain_cells = mesh.cell_block("tetra", physical_name="domain")
```

Важно: фильтрация идет именно по tag, а не по dimension.

Эквивалентная логика:

```python
tag = mesh.physical_tag("domain")
tags = mesh.cell_tags["tetra"]
domain_cells = mesh.cell_blocks["tetra"][tags == tag]
```

## Конвертация в `MeshData`

`to_mesh_data()` превращает один cell block в легкий контейнер `MeshData`.

```python
domain = mesh.to_mesh_data(
    cell_type="tetra",
    physical_name="domain",
)

assert domain.cell_type == "tetra"
assert domain.metadata["physical_name"] == "domain"
assert domain.metadata["physical_dimension"] == 3
assert domain.metadata["physical_tag"] == 1
```

Если `physical_name` не передан, в `MeshData` попадет весь блок выбранного `cell_type`.

Metadata результата содержит:

- `source`: `"MeshData"`;
- `cell_type`: выбранный тип ячеек;
- `physical_name`: имя группы или `None`;
- `field_data`: внутренний словарь physical groups;
- `physical_dimension`: только если передан `physical_name`;
- `physical_tag`: только если передан `physical_name`.

## Чтение `.msh`

```python
from geometry import read_gmsh_meshio

mesh = read_gmsh_meshio("torso.msh", dim=3)

print(mesh.field_data)
print(mesh.physical_dimension("domain"))
print(mesh.physical_tag("domain"))
```

Для текущей тестовой сетки ожидается:

```python
mesh.field_data["domain"] == (3, 1)
mesh.field_data["boundary"] == (2, 2)
mesh.physical_dimension("domain") == 3
mesh.physical_tag("domain") == 1
```

Получение объема и поверхности:

```python
volume_mesh = mesh.to_mesh_data(
    cell_type="tetra",
    physical_name="domain",
)

surface_mesh = mesh.to_mesh_data(
    cell_type="triangle",
    physical_name="boundary",
)
```

Не привязывайте production-код к конкретным counts из repository mesh files: они меняются при refinement и remeshing.

### Surface point arrays

`to_mesh_data("triangle", ...)` сохраняет исходный global point array и connectivity выбранных triangles. Поэтому `surface_mesh.num_points` может совпадать с `volume_mesh.num_points`, хотя surface triangles используют существенно меньше вершин:

```python
used_surface_vertex_ids = np.unique(surface_mesh.cells.ravel())
print("point array size:", surface_mesh.num_points)
print("used surface vertices:", used_surface_vertex_ids.size)
```

Такое представление сохраняет global node ids и не является ошибкой. Для памяти можно позднее добавить explicit compaction helper, но нельзя неявно перенумеровывать nodes там, где downstream metadata зависит от исходных ids.

## Типовой пайплайн

```python
import numpy as np

from geometry import (
    ElectrodeSet,
    SourceRegion,
    TorsoGeometry,
    read_gmsh_meshio,
    validate_torso_geometry,
)

mesh = read_gmsh_meshio("torso.msh", dim=3)

volume_mesh = mesh.to_mesh_data("tetra", physical_name="domain")
surface_mesh = mesh.to_mesh_data("triangle", physical_name="boundary")

electrodes = ElectrodeSet(
    positions=np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    ),
    labels=["E1", "E2"],
)

source_region = SourceRegion.from_bounding_box(
    mesh=volume_mesh,
    bounds_min=[-50.0, -50.0, -50.0],
    bounds_max=[50.0, 50.0, 50.0],
    mode="center",
)

geometry = TorsoGeometry(
    geometry_id="torso_from_msh",
    volume_mesh=volume_mesh,
    surface_mesh=surface_mesh,
    electrodes=electrodes,
    source_region=source_region,
)

report = validate_torso_geometry(geometry)
print(report.is_valid)
print(report.summary)
```

## Частые ошибки

Если `cell_block("tetra", physical_name="domain")` возвращает 0 ячеек, проверьте:

- что `field_data["domain"]` хранится как `(dim, tag)`, например `(3, 1)`;
- что `cell_tags["tetra"]` содержит physical tag, например `1`;
- что данные пришли через `read_gmsh_meshio()`, если исходный источник это `meshio`;
- что вы не передали вручную meshio-порядок `(tag, dim)` в `MeshData`.

Если нужна вся сетка без фильтрации physical group:

```python
all_tetra = mesh.cell_block("tetra")
all_volume = mesh.to_mesh_data("tetra")
```
