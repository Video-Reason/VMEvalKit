# VMEvalKit 🎥🧠

Evaluate reasoning capabilities in video generation models through cognitive tasks.

## Overview

VMEvalKit tests whether video models can solve visual problems (mazes, chess, puzzles) by generating solution videos. 

**Key requirement**: Models must accept BOTH:
- 📸 An input image (the problem)
- 📝 A text prompt (instructions)

## Installation

```bash
git clone https://github.com/yourusername/VMEvalKit.git
cd VMEvalKit
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from vmevalkit import run_inference

# Generate video solution
result = run_inference(
    model_name="luma-ray-2",
    image_path="data/maze.png",
    text_prompt="Solve this maze from start to finish"
)

print(f"Video: {result['video_path']}")
```

## Supported Models

| Model | Name | Notes |
|-------|------|-------|
| **Luma Ray 2** | `luma-ray-2` | High quality |
| **Luma Ray Flash 2** | `luma-ray-flash-2` | Faster generation |

Both require `LUMA_API_KEY` in environment.

## Tasks

- **Maze Solving**: Navigate from start to finish
- **Mental Rotation**: Rotate 3D objects to match targets
- **Chess Puzzles**: Demonstrate puzzle solutions
- **Raven's Matrices**: Complete visual patterns

## Configuration

Create `.env`:
```bash
LUMA_API_KEY=your_key_here
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET=vmevalkit
AWS_DEFAULT_REGION=us-east-2
```

## Project Structure

```
VMEvalKit/
├── vmevalkit/
│   ├── runner/         # Inference runners
│   ├── models/         # Model implementations
│   ├── core/           # Evaluation framework
│   ├── tasks/          # Task definitions
│   └── utils/          # Utilities
├── data/               # Datasets
├── examples/           # Example scripts
└── tests/              # Unit tests
```

## Examples

See `examples/simple_inference.py` for more usage patterns.

## Submodules

Initialize after cloning:
```bash
git submodule update --init --recursive
```

- **KnowWhat**: Research on knowing-how vs knowing-that
- **maze-dataset**: Maze datasets for ML evaluation

## License

MIT