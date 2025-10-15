import random
from typing import List

from .rules import Rule_Wrapper


def sample_rules(max_components: int = 1) -> List[list]:
    """Minimal sampler: one component with Number/Position + one of Type/Size/Color.
    Returns list[component][rules]. The first rule is for Number/Position to mimic priority.
    """
    component_idx = 0
    # choose number vs position for the main rule
    main_attr = random.choice(["Number", "Position"])
    step = random.choice([-1, 1])
    rules = [Rule_Wrapper("Progression", main_attr, step, component_idx)]

    # choose one additional attribute rule
    extra_attr = random.choice(["Type", "Size", "Color"]) 
    rules.append(Rule_Wrapper("Progression", extra_attr, 1, component_idx))
    return [rules]


