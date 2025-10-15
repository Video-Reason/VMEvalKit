import copy
from typing import List

from .aot import Root


class Rule:
    def __init__(self, attr: str, component_idx: int, step: int = 1):
        self.attr = attr  # "Number" | "Position" | "Type" | "Size" | "Color"
        self.component_idx = component_idx
        self.step = step

    def apply_rule(self, aot: Root, in_aot: Root = None) -> Root:
        raise NotImplementedError


class Constant(Rule):
    def apply_rule(self, aot: Root, in_aot: Root = None) -> Root:
        return copy.deepcopy(in_aot or aot)


class Progression(Rule):
    def apply_rule(self, aot: Root, in_aot: Root = None) -> Root:
        base = aot
        out = copy.deepcopy(in_aot or aot)
        base_layout = base.children[0].components[self.component_idx].layout
        out_layout = out.children[0].components[self.component_idx].layout

        if self.attr == "Number":
            count = len(out_layout.entities)
            out_layout.set_num(max(0, min(count + self.step, len(out_layout.positions))))
        elif self.attr == "Position":
            out_layout.set_positions_by_offset(self.step)
        elif self.attr == "Type":
            for ent in out_layout.entities:
                ent.type.set_idx(ent.type.get_idx() + self.step)
        elif self.attr == "Size":
            for ent in out_layout.entities:
                ent.size.set_idx(ent.size.get_idx() + self.step)
        elif self.attr == "Color":
            for ent in out_layout.entities:
                ent.color.set_idx(ent.color.get_idx() + self.step)
        else:
            raise ValueError("Unsupported attribute for Progression")
        return out


def Rule_Wrapper(name: str, attr: str, param, component_idx: int):
    if name == "Constant":
        return Constant(attr, component_idx, step=0)
    if name == "Progression":
        step = 1 if not param else (param if isinstance(param, int) else 1)
        return Progression(attr, component_idx, step=step)
    # Fallback: map complex rules to Constant for minimal viable generation
    return Constant(attr, component_idx, step=0)


