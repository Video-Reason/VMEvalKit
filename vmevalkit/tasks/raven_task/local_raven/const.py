IMAGE_SIZE = 160

# Drawing defaults
DEFAULT_LINE_WIDTH = 2

# Discrete levels for attributes (indexes, not direct values)
TYPE_VALUES = ["triangle", "square", "pentagon", "hexagon", "circle"]
SIZE_VALUES = [0.45, 0.55, 0.65, 0.75, 0.85]
COLOR_VALUES = [0, 28, 56, 84, 112, 140, 168, 196, 224, 255]

# Convenience clamps
def clamp(value, lo, hi):
    return max(lo, min(hi, value))


