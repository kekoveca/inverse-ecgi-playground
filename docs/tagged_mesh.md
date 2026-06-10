# TaggedMesh: Gmsh/meshio сетки с physical groups

`geometry.tagged_mesh` хранит сетки, прочитанные из Gmsh через `meshio`, вместе с physical groups. Этот слой нужен, когда в одном файле лежат разные блоки ячеек: например `tetra` для объема торса и `triangle` для поверхности.

Для downstream-кода, которому нужен один конкретный блок, `TaggedMesh` преобразуется в обычный `MeshData`.

## Основные сущности

- `TaggedMesh`: контейнер для координат, cell blocks, physical tags и metadata.
- `Mesh`: обратносовместимый alias для `TaggedMesh`.
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

```python
# meshio convention
name -> (tag, dim)
```

Поэтому `read_gmsh_meshio()` вызывает `_field_data_to_tuples()` и переворачивает пары:

```python
meshio:   (tag, dim)
internal: (dim, tag)
```

Например:

```python
from geometry.tagged_mesh import _field_data_to_tuples
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

Если вы создаете `TaggedMesh` вручную, передавайте уже внутренний порядок `(dim, tag)`.

## Структура `TaggedMesh`

```python
from geometry import Mesh
import numpy as np

tagged = Mesh(
    dim=3,
    coords=np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    cells={
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
- `coords`: массив узлов формы `(n_nodes, dim)`.
- `cells`: словарь `cell_type -> connectivity`, например `"tetra"`, `"triangle"`, `"line"`.
- `cell_tags`: словарь `cell_type -> physical tag per cell`.
- `field_data`: словарь `physical_name -> (dim, tag)`.
- `metadata`: произвольные метаданные.

Если для блока ячеек нет `cell_tags`, контейнер создаст нулевые tags для всех ячеек этого блока.

## Physical group API

```python
assert tagged.physical_dimension("domain") == 3
assert tagged.physical_tag("domain") == 1
```

`physical_dimension(name)` возвращает первый элемент `field_data[name]`.

`physical_tag(name)` возвращает второй элемент `field_data[name]`.

Если physical group неизвестна, методы бросают `KeyError` со списком доступных групп.

## Фильтрация cell blocks

`cell_block(cell_type, physical_name=None)` возвращает connectivity-массив для указанного типа ячеек. Если передан `physical_name`, ячейки фильтруются по physical tag:

```python
domain_cells = tagged.cell_block("tetra", physical_name="domain")
```

Важно: фильтрация идет именно по tag, а не по dimension.

Эквивалентная логика:

```python
tag = tagged.physical_tag("domain")
tags = tagged.cell_tags["tetra"]
domain_cells = tagged.cells["tetra"][tags == tag]
```

## Конвертация в `MeshData`

`to_mesh_data()` превращает один cell block в легкий контейнер `MeshData`.

```python
domain = tagged.to_mesh_data(
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

- `source`: `"TaggedMesh"`;
- `cell_type`: выбранный тип ячеек;
- `physical_name`: имя группы или `None`;
- `field_data`: внутренний словарь physical groups;
- `physical_dimension`: только если передан `physical_name`;
- `physical_tag`: только если передан `physical_name`.

## Чтение `.msh`

```python
from geometry import read_gmsh_meshio

tagged = read_gmsh_meshio("examples/torso.msh", dim=3)

print(tagged.field_data)
print(tagged.physical_dimension("domain"))
print(tagged.physical_tag("domain"))
```

Для текущей тестовой сетки ожидается:

```python
tagged.field_data["domain"] == (3, 1)
tagged.field_data["boundary"] == (2, 2)
tagged.physical_dimension("domain") == 3
tagged.physical_tag("domain") == 1
```

Получение объема и поверхности:

```python
volume_mesh = tagged.to_mesh_data(
    cell_type="tetra",
    physical_name="domain",
)

surface_mesh = tagged.to_mesh_data(
    cell_type="triangle",
    physical_name="boundary",
)
```

Для текущей `examples/torso.msh`:

```python
volume_mesh.num_cells == 47158
surface_mesh.num_cells == 7760
```

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

tagged = read_gmsh_meshio("examples/torso.msh", dim=3)

volume_mesh = tagged.to_mesh_data("tetra", physical_name="domain")
surface_mesh = tagged.to_mesh_data("triangle", physical_name="boundary")

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
- что вы не передали вручную meshio-порядок `(tag, dim)` в `TaggedMesh`.

Если нужна вся сетка без фильтрации physical group:

```python
all_tetra = tagged.cell_block("tetra")
all_volume = tagged.to_mesh_data("tetra")
```
