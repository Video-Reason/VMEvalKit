import copy
import random
from typing import List, Tuple

from .const import TYPE_VALUES, SIZE_VALUES, COLOR_VALUES


class Attribute:
    def __init__(self, levels: List):
        self.levels = levels
        self.idx = random.randrange(len(levels))

    def set_idx(self, idx: int):
        self.idx = max(0, min(idx, len(self.levels) - 1))

    def get_idx(self) -> int:
        return self.idx

    def get_value(self):
        return self.levels[self.idx]


class Entity:
    def __init__(self, bbox: Tuple[float, float, float, float]):
        self.bbox = bbox  # (y, x, h, w) in relative coords
        self.type = Attribute(TYPE_VALUES)
        self.size = Attribute(SIZE_VALUES)
        self.color = Attribute(COLOR_VALUES)

    def clone(self):
        e = Entity(self.bbox)
        e.type.set_idx(self.type.get_idx())
        e.size.set_idx(self.size.get_idx())
        e.color.set_idx(self.color.get_idx())
        return e


class Layout:
    def __init__(self, positions: List[Tuple[float, float, float, float]]):
        self.positions = positions
        self.entities: List[Entity] = []
        # initialize with 1 entity if available
        if positions:
            self.entities = [Entity(positions[0])]

    def set_num(self, n: int):
        n = max(0, min(n, len(self.positions)))
        cur = len(self.entities)
        if n > cur:
            for i in range(cur, n):
                self.entities.append(Entity(self.positions[i]))
        elif n < cur:
            self.entities = self.entities[:n]

    def set_positions_by_offset(self, offset: int):
        if not self.entities:
            return
        k = len(self.positions)
        for i, ent in enumerate(self.entities):
            ent.bbox = self.positions[(i + offset) % k]

    def clone(self):
        l = Layout(self.positions)
        l.entities = [e.clone() for e in self.entities]
        return l


class Component:
    def __init__(self, layout: Layout):
        self.layout = layout

    def clone(self):
        return Component(self.layout.clone())


class Structure:
    def __init__(self, components: List[Component]):
        self.components = components

    def clone(self):
        return Structure([c.clone() for c in self.components])


class Root:
    def __init__(self, structure: Structure):
        self.structure = structure

    def sample(self):
        # Minimal API parity: return a cloned PG tree
        return Root(self.structure.clone())

    def resample(self, change_number: bool = False):
        # Minimal stub: no stochastic resampling needed beyond sample()
        return

    def clone(self):
        return Root(self.structure.clone())

    # Compatibility with renderer expectations in our code
    @property
    def children(self):
        # mimic submodule: Root -> [Structure]
        return [self.structure]


