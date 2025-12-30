# ARC (Abstraction and Reasoning Corpus) Task

## 📋 Description

ARC (Abstraction and Reasoning Corpus) is a benchmark designed to measure abstract reasoning and generalization capabilities. This task module generates various grid-based transformation puzzles that require understanding and applying visual patterns.

## 🎯 Task Types

The ARC task generator includes the following transformation types:

| Type | Description | Difficulty |
|------|-------------|------------|
| **Color Swap** | Swap two specific colors in the grid | Easy |
| **Fill Pattern** | Fill enclosed regions with a specific color | Medium |
| **Mirror** | Create horizontal/vertical mirror of pattern | Easy |
| **Rotate** | Rotate the pattern 90 degrees | Medium |
| **Scale** | Scale the pattern by 2x | Medium |
| **Translate** | Move shapes to new positions | Easy |
| **Count and Fill** | Count objects and create corresponding output | Hard |
| **Border** | Add border around existing shapes | Medium |
| **Complete Pattern** | Fill in missing parts of repeating pattern | Hard |
| **Extract Shape** | Extract specific colored shape from noise | Medium |

## 🎨 Color Palette

ARC uses a standard 10-color palette:

| Index | Color | RGB |
|-------|-------|-----|
| 0 | Black (background) | (0, 0, 0) |
| 1 | Blue | (0, 116, 217) |
| 2 | Red | (255, 65, 54) |
| 3 | Green | (46, 204, 64) |
| 4 | Yellow | (255, 220, 0) |
| 5 | Gray | (170, 170, 170) |
| 6 | Magenta | (240, 18, 190) |
| 7 | Orange | (255, 133, 27) |
| 8 | Light Blue | (127, 219, 255) |
| 9 | Maroon | (135, 12, 37) |

## 📊 Data Format

Each task pair contains:
- `first_frame.png`: Input grid visualization
- `final_frame.png`: Expected output grid visualization
- `prompt.txt`: Task instructions
- `question_metadata.json`: Task metadata including:
  - `task_type`: Type of transformation
  - `description`: Human-readable description
  - `input_grid`: 2D array of color indices
  - `output_grid`: 2D array of color indices
  - `difficulty`: easy/medium/hard

## 🚀 Usage

### Generate Dataset

```bash
# Generate 50 ARC tasks
python examples/create_questions.py --task arc --pairs-per-domain 50

# Generate with specific count
python -m vmevalkit.runner.create_dataset --task arc --pairs-per-domain 100
```

### Test Locally

```python
from vmevalkit.tasks.arc_task import create_dataset

# Generate 5 test tasks
dataset = create_dataset(num_samples=5)
print(f"Generated {len(dataset['pairs'])} tasks")
```

## ⚖️ Evaluation Criteria

When scoring ARC task completions:

1. **Grid Accuracy** (0-10): Does the output match the expected transformation?
2. **Color Correctness**: Are the correct colors in the correct positions?
3. **Pattern Understanding**: Was the transformation rule correctly inferred?

## 📚 References

- [ARC Prize](https://arcprize.org/)
- [Original ARC Paper](https://arxiv.org/abs/1911.01547)
- [ARC Dataset on GitHub](https://github.com/fchollet/ARC)

## 📝 Notes

- Grid sizes range from 3x3 to 12x12
- Tasks are procedurally generated for unlimited variations
- Each task has a single correct solution
- Difficulty is automatically assigned based on task complexity
