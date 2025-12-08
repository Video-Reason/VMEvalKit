# Dot to Dot Task Scaling Plan

## Overview
This document outlines the scaling strategy for the dot-to-dot puzzle task, designed to create a diverse dataset with varying difficulty levels and complexity.

## Scaling Dimensions

### 1. Number of Dots (Primary Difficulty Factor)

| Difficulty | Dot Range | Description | Use Cases |
|------------|-----------|-------------|-----------|
| **easy** | 5-10 | Simple patterns, clear shapes | Entry-level reasoning |
| **medium** | 10-15 | Moderate complexity | Standard evaluation |
| **hard** | 15-20 | Complex patterns (max 20 dots) | Advanced reasoning |

**Note:** Maximum dots is capped at 20 for all difficulty levels.

**Distribution Strategy:**
- 40% easy (5-10 dots)
- 35% medium (10-15 dots)
- 25% hard (15-20 dots)

### 2. Pattern Types (Complexity-Based Selection)

**Simple Patterns** (Geometric, predictable):
- `circle` - Uniform circular distribution
- `triangle` - Three-sided polygon
- `square` - Four-sided polygon
- `pentagon` - Five-sided polygon (new)
- `hexagon` - Six-sided polygon (new)

**Moderate Patterns** (Curved, recognizable):
- `star` - Star shape with alternating radii
- `heart` - Heart shape using parametric equations
- `diamond` - Diamond/rhombus shape (new)
- `oval` - Elliptical shape (new)

**Complex Patterns** (Non-linear, abstract):
- `spiral` - Archimedean spiral
- `wave` - Sine wave pattern (new)
- `zigzag` - Zigzag pattern (new)
- `flower` - Flower/petal pattern (new)

**Pattern Selection by Difficulty:**
- **very_easy/easy**: Simple patterns only (circle, triangle, square)
- **medium**: Simple + Moderate (add star, heart, diamond)
- **hard**: All patterns except most complex
- **very_hard**: All patterns including complex ones (spiral, wave, flower)

### 3. Pattern Size (Canvas Coverage)

| Size | Coverage | Margin | Use Case |
|------|----------|--------|----------|
| **small** | 30-40% | 120px | Compact patterns, more space around |
| **medium** | 50-60% | 80px | Standard size (current default) |
| **large** | 70-80% | 50px | Large patterns, fills canvas |

**Size Selection:**
- Easy patterns: Prefer medium/large (easier to see)
- Hard patterns: Mix of all sizes (adds variety)

### 4. Point Distribution Density

| Density | Min Distance | Description |
|---------|--------------|-------------|
| **sparse** | DOT_RADIUS * 4 | More space between dots |
| **normal** | DOT_RADIUS * 3 | Standard spacing (current) |
| **dense** | DOT_RADIUS * 2.5 | Closer dots, more challenging |

**Density Selection:**
- Easy: Prefer sparse/normal
- Hard: Can use dense for added challenge

### 5. Pattern Orientation & Variation

**Orientation:**
- Standard (0°)
- Rotated (45°, 90°, 135°, 180°)
- Random rotation (0-360°)

**Variation:**
- Standard pattern
- Stretched pattern (non-uniform scaling)
- Skewed pattern (slight distortion)

## Implementation Strategy

### Phase 1: Basic Scaling (Current)
- ✅ 3 difficulty levels (easy, medium, hard)
- ✅ 6 pattern types
- ✅ Fixed pattern size
- ✅ Normal density

### Phase 2: Enhanced Scaling
- [ ] Add very_easy and very_hard difficulties
- [ ] Add 4 new pattern types (pentagon, hexagon, diamond, oval)
- [ ] Implement pattern size variation
- [ ] Add density control
- [ ] Add orientation rotation

### Phase 3: Advanced Scaling
- [ ] Add complex patterns (wave, zigzag, flower)
- [ ] Implement pattern stretching/skewing
- [ ] Add pattern combination (e.g., star inside circle)
- [ ] Variable dot sizes (for very_hard)

## Scaling Distribution Example

For 100 questions:

```
very_easy (20 questions):
  - 5-8 dots
  - Simple patterns: circle (40%), triangle (30%), square (30%)
  - Medium/large size
  - Sparse/normal density

easy (30 questions):
  - 8-12 dots
  - Simple patterns: circle, triangle, square, pentagon
  - Medium size
  - Normal density

medium (30 questions):
  - 15-25 dots
  - Simple + Moderate: star (25%), heart (20%), circle (20%), triangle (15%), square (10%), diamond (10%)
  - Medium size
  - Normal density

hard (15 questions):
  - 30-50 dots
  - All patterns except most complex
  - Mix of sizes
  - Normal/dense density

very_hard (5 questions):
  - 50-80 dots
  - Complex patterns: spiral, wave, flower
  - Large size
  - Dense density
```

## Code Structure Changes

### New Difficulty Levels
```python
def _dots_for_difficulty(self, difficulty: str) -> int:
    if difficulty == "very_easy":
        return self.rng.randint(5, 8)
    if difficulty == "easy":
        return self.rng.randint(8, 12)
    if difficulty == "medium":
        return self.rng.randint(15, 25)
    if difficulty == "hard":
        return self.rng.randint(30, 50)
    if difficulty == "very_hard":
        return self.rng.randint(50, 80)
    return self.rng.randint(15, 25)  # default to medium
```

### Pattern Selection by Difficulty
```python
def _select_pattern_type(self, num_dots: int, difficulty: str) -> str:
    if difficulty in ["very_easy", "easy"]:
        return self.rng.choice(["circle", "triangle", "square", "pentagon"])
    elif difficulty == "medium":
        return self.rng.choice(["circle", "triangle", "square", "star", "heart", "diamond"])
    elif difficulty == "hard":
        return self.rng.choice([
            "circle", "triangle", "square", "star", "heart", 
            "diamond", "oval", "spiral"
        ])
    else:  # very_hard
        return self.rng.choice([
            "star", "heart", "spiral", "wave", "zigzag", "flower"
        ])
```

### Pattern Size Variation
```python
def _get_pattern_size(self, difficulty: str) -> Tuple[str, int]:
    """Returns (size_type, margin)"""
    if difficulty in ["very_easy", "easy"]:
        size_type = self.rng.choice(["medium", "large"])
        margin = 80 if size_type == "medium" else 50
    elif difficulty == "medium":
        size_type = "medium"
        margin = 80
    else:  # hard, very_hard
        size_type = self.rng.choice(["small", "medium", "large"])
        margin = {"small": 120, "medium": 80, "large": 50}[size_type]
    return size_type, margin
```

## Quality Assurance

### Uniqueness Requirements
- Each pattern type + dot count + size combination should be unique
- Minimum distance between dots enforced
- Sequential numbering (1, 2, 3, ...) always maintained

### Validation Checks
- [ ] All dots have unique positions
- [ ] All dots have sequential numbers
- [ ] Pattern is recognizable when connected
- [ ] No overlapping dots
- [ ] Points are within canvas bounds
- [ ] Final pattern matches expected shape

## Metrics to Track

1. **Pattern Distribution**: Count of each pattern type
2. **Difficulty Distribution**: Count per difficulty level
3. **Dot Count Distribution**: Histogram of dot counts
4. **Pattern Size Distribution**: small/medium/large ratios
5. **Uniqueness Rate**: Percentage of unique task signatures

## Future Enhancements

1. **Multi-pattern Tasks**: Connect multiple separate patterns
2. **Hidden Patterns**: Patterns that aren't obvious until connected
3. **Color-coded Dots**: Different colors for different segments
4. **Variable Dot Sizes**: Larger dots for key points
5. **Animated Hints**: Subtle visual hints in first frame

