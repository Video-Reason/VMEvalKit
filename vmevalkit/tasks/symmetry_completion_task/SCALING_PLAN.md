# Symmetry Completion Task - Scaling Plan

## Overview

This document outlines the scaling strategy for the **Symmetry Completion Task**, which tests models' ability to complete visual patterns by recognizing and applying left-right symmetry.

## Core Task Design

- **Final Frame**: Complete left-right symmetric pattern
- **First Frame**: Left half visible, right half missing (to be completed)
- **Model Task**: Infer right half by mirroring left half across vertical axis
- **Grid Constraint**: All grids use even sizes (4, 6, 8, 10, ...) for perfect symmetry

## Scaling Dimensions

### 1. Grid Size (Block Count) - Primary Difficulty Factor

| Difficulty | Grid Sizes | Total Blocks | Reasoning Complexity |
|------------|-----------|--------------|---------------------|
| **Easy** | 4×4, 6×6 | 16-36 | Low - Fewer cells to reason about |
| **Medium** | 6×6, 8×8 | 36-64 | Moderate - More cells, more patterns |
| **Hard** | 8×8, 10×10 | 64-100 | High - Many cells, complex patterns |
| **Very Hard** | 12×12, 14×14 | 144-196 | Very High - Large grids, complex reasoning |

**Scaling Logic:**
- Larger grids = more cells to complete = more reasoning steps
- More blocks = more pattern complexity to recognize
- Grid size directly correlates with cognitive load

### 2. Missing Cell Ratio (Right Half) - Secondary Difficulty Factor

| Difficulty | Missing Ratio | Visible in Right Half | Reasoning Challenge |
|------------|---------------|----------------------|---------------------|
| **Easy** | 30-40% | 60-70% visible | Low - Most pattern visible |
| **Medium** | 50-60% | 40-50% visible | Moderate - Half pattern visible |
| **Hard** | 70-80% | 20-30% visible | High - Little pattern visible |
| **Very Hard** | 85-95% | 5-15% visible | Very High - Minimal pattern visible |

**Scaling Logic:**
- More missing cells = less information = harder to infer pattern
- Lower visible ratio = requires stronger pattern recognition
- Missing ratio creates fine-grained difficulty control

### 3. Pattern Type Complexity - Tertiary Difficulty Factor

| Pattern Type | Complexity | Recognition Difficulty | Example |
|--------------|-----------|------------------------|---------|
| **vertical_symmetry_stripes** | Low | Easy - Simple alternating pattern | Horizontal stripes |
| **vertical_symmetry_checkerboard** | Low-Medium | Moderate - Alternating grid pattern | Chessboard |
| **vertical_symmetry** | Medium | Moderate - Random but symmetric | Random symmetric |
| **vertical_symmetry_increment** | Medium-High | Hard - Progressive pattern | Row increment |

**Future Pattern Types (Potential):**
- **Nested Symmetry**: Symmetry within symmetry
- **Multi-layer Symmetry**: Multiple symmetric layers
- **Combined Patterns**: Checkerboard + stripes, etc.
- **Asymmetric Base**: Complex base pattern with symmetry

## Current Implementation

### Difficulty Mapping

```python
difficulty_map = {
    "easy": {
        "grid_sizes": [4, 6],           # 16-36 blocks
        "missing_ratio": (0.3, 0.4),    # 30-40% missing
        "pattern_types": ["vertical_symmetry", "vertical_symmetry_checkerboard", 
                         "vertical_symmetry_stripes", "vertical_symmetry_increment"]
    },
    "medium": {
        "grid_sizes": [6, 8],           # 36-64 blocks
        "missing_ratio": (0.5, 0.6),    # 50-60% missing
        "pattern_types": ["vertical_symmetry", "vertical_symmetry_checkerboard", 
                         "vertical_symmetry_stripes", "vertical_symmetry_increment"]
    },
    "hard": {
        "grid_sizes": [8, 10],          # 64-100 blocks
        "missing_ratio": (0.7, 0.8),    # 70-80% missing
        "pattern_types": ["vertical_symmetry", "vertical_symmetry_checkerboard", 
                         "vertical_symmetry_stripes", "vertical_symmetry_increment"]
    }
}
```

## Scaling Strategy

### Phase 1: Current Implementation (50+ samples)

**Grid Sizes**: 4×4, 6×6, 8×8, 10×10
**Missing Ratios**: 30-40% (easy), 50-60% (medium), 70-80% (hard)
**Pattern Types**: 4 types
**Total Combinations**: ~200+ unique combinations
**Sample Capacity**: 50-100 samples

### Phase 2: Extended Grid Sizes (100+ samples)

**Add**: 12×12, 14×14 grids
**Grid Sizes**: 4×4, 6×6, 8×8, 10×10, 12×12, 14×14
**Missing Ratios**: Same as Phase 1
**Pattern Types**: Same as Phase 1
**Total Combinations**: ~400+ unique combinations
**Sample Capacity**: 100-200 samples

### Phase 3: Fine-grained Missing Ratios (200+ samples)

**Add**: More granular missing ratios
**Missing Ratios**: 
- Easy: 20%, 30%, 40%
- Medium: 45%, 50%, 55%, 60%
- Hard: 65%, 70%, 75%, 80%
- Very Hard: 85%, 90%, 95%

**Total Combinations**: ~800+ unique combinations
**Sample Capacity**: 200-500 samples

### Phase 4: Advanced Pattern Types (500+ samples)

**Add**: Complex pattern types
- Nested symmetry patterns
- Multi-layer patterns
- Combined patterns (checkerboard + stripes, etc.)

**Total Combinations**: ~2000+ unique combinations
**Sample Capacity**: 500-1000+ samples

## Uniqueness Guarantee

Each task is uniquely identified by:
- Pattern type
- Grid size
- Missing positions (right half)
- Full pattern structure

**Signature**: `{pattern_type}-{grid_size}-{missing_positions}-{pattern_hash}`

Since left half is always fully visible, right half is uniquely determined by symmetry, ensuring:
- **Solution Uniqueness**: Each first_frame has exactly one correct final_frame
- **Task Uniqueness**: Different missing positions create different tasks

## Difficulty Progression Examples

### Easy → Medium → Hard

**Example 1: Checkerboard Pattern**
- Easy: 4×4 grid, 30% missing (right half) → 3 cells missing
- Medium: 6×6 grid, 50% missing (right half) → 9 cells missing
- Hard: 10×10 grid, 75% missing (right half) → 38 cells missing

**Example 2: Increment Pattern**
- Easy: 4×4 grid, 35% missing → Simple increment visible
- Medium: 8×8 grid, 55% missing → Moderate increment pattern
- Hard: 10×10 grid, 80% missing → Complex increment with minimal info

## Scaling Metrics

### Current Capacity
- **Pattern Types**: 4
- **Grid Sizes**: 4 (4, 6, 8, 10)
- **Missing Ratios**: 3 ranges (easy, medium, hard)
- **Total Combinations**: ~200+
- **Generated Samples**: 50

### Potential Capacity
- **Pattern Types**: 8+ (with future additions)
- **Grid Sizes**: 6+ (4, 6, 8, 10, 12, 14)
- **Missing Ratios**: 10+ (fine-grained)
- **Total Combinations**: 2000+
- **Potential Samples**: 1000+

## Implementation Notes

1. **Even Grid Sizes Only**: All grids must be even (4, 6, 8, 10, ...) for perfect symmetry
2. **Right Half Only**: Missing cells are always from the right half
3. **Symmetry Verification**: All generated patterns are verified for perfect left-right symmetry
4. **Unique Solution**: Left half fully visible ensures unique solution

## Future Enhancements

1. **Horizontal Symmetry**: Extend to top-bottom symmetry
2. **Diagonal Symmetry**: Add diagonal symmetry patterns
3. **Multiple Symmetry Axes**: Patterns with multiple symmetry axes
4. **Asymmetric Completion**: Complete asymmetric patterns (not just symmetric)
5. **3D Symmetry**: Extend to 3D grid symmetry (future work)

