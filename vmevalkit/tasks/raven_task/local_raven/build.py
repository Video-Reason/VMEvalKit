from typing import List, Tuple

from .aot import Root, Structure, Component, Layout


def _grid_positions(grid_rows: int, grid_cols: int) -> List[Tuple[float, float, float, float]]:
    positions = []
    cell_h = 1.0 / grid_rows
    cell_w = 1.0 / grid_cols
    offsets = [i + 0.5 for i in range(grid_rows)]
    for r in range(grid_rows):
        for c in range(grid_cols):
            y = (r + 0.5) * cell_h
            x = (c + 0.5) * cell_w
            positions.append((y, x, min(cell_h, cell_w) * 0.9, min(cell_h, cell_w) * 0.9))
    return positions


def build_center_single() -> Root:
    positions = [(0.5, 0.5, 1.0, 1.0)]
    layout = Layout(positions)
    comp = Component(layout)
    struct = Structure([comp])
    return Root(struct)


def build_distribute_four() -> Root:
    positions = _grid_positions(2, 2)
    layout = Layout(positions)
    comp = Component(layout)
    struct = Structure([comp])
    return Root(struct)


def build_distribute_nine() -> Root:
    positions = _grid_positions(3, 3)
    layout = Layout(positions)
    comp = Component(layout)
    struct = Structure([comp])
    return Root(struct)


