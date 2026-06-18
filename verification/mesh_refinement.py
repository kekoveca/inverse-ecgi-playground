from __future__ import annotations

import numpy as np

from geometry import MeshData


def create_unit_cube_meshdata(n: int) -> MeshData:
    """Create a conforming structured tetrahedral mesh of ``[0, 1]^3``.

    Each of the ``n^3`` cubes is split into six tetrahedra around its main
    diagonal. The returned mesh is independent of DOLFINx.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be a positive integer")

    axis = np.linspace(0.0, 1.0, n + 1)
    points = np.array([(x, y, z) for x in axis for y in axis for z in axis], dtype=float)
    stride_y = n + 1
    stride_x = (n + 1) ** 2

    def node(i: int, j: int, k: int) -> int:
        return i * stride_x + j * stride_y + k

    cells: list[list[int]] = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v000 = node(i, j, k)
                v100 = node(i + 1, j, k)
                v010 = node(i, j + 1, k)
                v110 = node(i + 1, j + 1, k)
                v001 = node(i, j, k + 1)
                v101 = node(i + 1, j, k + 1)
                v011 = node(i, j + 1, k + 1)
                v111 = node(i + 1, j + 1, k + 1)
                cells.extend(
                    [
                        [v000, v100, v110, v111],
                        [v000, v110, v010, v111],
                        [v000, v010, v011, v111],
                        [v000, v011, v001, v111],
                        [v000, v001, v101, v111],
                        [v000, v101, v100, v111],
                    ]
                )

    return MeshData(
        points=points,
        cells=np.asarray(cells, dtype=np.int64),
        cell_type="tetra",
        name=f"unit_cube_n{n}",
        metadata={"domain": "unit_cube", "subdivisions": n, "h": 1.0 / n},
    )
