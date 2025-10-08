# VMEvalKit 🎥🧠

**VMEvalKit** is a comprehensive evaluation framework for assessing reasoning capabilities in video generation models through cognitive and logical tasks.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

VMEvalKit addresses a critical gap in evaluating modern video generation models beyond visual quality metrics. While existing benchmarks focus on photorealism and temporal consistency, VMEvalKit evaluates whether video models can demonstrate genuine reasoning capabilities by solving visual problems through generated video sequences.

### 🔴 Critical Requirement: Text + Image → Video

**VMEvalKit specifically requires models that accept BOTH:**
- 📸 **An input image** (the problem to solve: maze, chess board, etc.)
- 📝 **A text prompt** (instructions: "solve the maze", "show the next move", etc.)

Many models advertise "image-to-video" but only accept images WITHOUT text guidance. These are NOT suitable for reasoning evaluation as they cannot receive problem-solving instructions.

Our framework tests these specialized text+image→video models on fundamental reasoning tasks, requiring them to:
- Understand complex visual problems from the input image
- Follow text instructions to demonstrate solutions
- Generate videos that show logical problem-solving steps

## ✨ Key Features

- **🧩 Diverse Reasoning Tasks**: Evaluate models on maze solving, mental rotation, chess positions, and Raven's Progressive Matrices
- **🔌 Unified Interface**: Support for both open-source and closed-source video generation models
- **📊 Comprehensive Metrics**: Automatic evaluation of reasoning accuracy, solution correctness, and video quality
- **🚀 Easy Integration**: Simple API for adding new models and tasks
- **📈 Benchmarking Suite**: Compare performance across multiple models and tasks
- **🎨 Visualization Tools**: Generate detailed reports with video outputs and analysis

## 📋 Supported Tasks

### 1. **Maze Solving** 🌀
Tests spatial reasoning and pathfinding abilities. Models receive a maze image and must generate a video showing the solution path from start to finish.

### 2. **Mental Rotation** 🔄
Evaluates 3D spatial understanding. Models must generate videos showing objects rotating to match target orientations.

### 3. **Chess Puzzles** ♟️
Assesses strategic reasoning. Models generate videos demonstrating chess puzzle solutions with legal moves.

### 4. **Raven's Progressive Matrices** 🔲
Tests abstract pattern recognition. Models complete visual patterns by generating the missing sequence element.

## 🤖 Supported Models

VMEvalKit focuses on evaluating models that can accept BOTH a still image AND a text prompt to generate videos. This is a specialized capability that not all video generation models possess.

### ✅ Text+Image→Video Models

These closed-source APIs offer the capability to process both text prompts AND image inputs simultaneously:

#### Closed-Source APIs

##### ⚠️ Runway API Models (Limited Support)
Based on the [official Runway API documentation](https://docs.dev.runwayml.com/guides/models/):
- **gen4_turbo** - Image → Video only (no text prompt support)
- **gen4_aleph** - Video + Text/Image → Video (requires video input, not suitable)
- **act_two** - Image or Video → Video (no text prompt support mentioned)
- **veo3** - Text OR Image → Video (not both simultaneously)

**Note**: Current Runway API models do NOT support the text+image→video capability required for reasoning tasks.

##### ✅ Other Verified APIs
- **Pika 2.0+** - Supports image+prompt for guided video generation
- **Google Imagen Video** - Cascade model supporting text+image inputs
- **Luma Dream Machine** - Offers image reference with text prompt guidance
- **Stability AI Video** - Enterprise API with text-conditioned image animation
- **Genmo Mochi** - Multimodal video generation with text and image inputs

### ⚠️ Image-Only Models (No Text Conditioning)

These models only accept images WITHOUT text prompts - not suitable for reasoning evaluation:
- **Stable Video Diffusion (SVD)** - Image-only input, no text prompt support
- **AnimateDiff** (basic version) - Image animation without text guidance

### ❓ Additional Models Requiring Verification

These commercial models may offer text+image capabilities but require verification:
- **WaveSpeed Wan 2.2** - Multiple modes including i2v and t2v, possible combined support
- **Kling** (Kuaishou) - Advanced I2V model, text prompt support needs verification
- **Haiper** - Has I2V endpoint, text conditioning capabilities need confirmation
- **MiniMax Hailuo** - Chinese model with I2V capability, text support verification needed
- **Leonardo.ai Motion** - Creative platform with potential text+image video generation
- **Synthesia** - Enterprise video platform, API capabilities need verification

### 🔍 Important Notes on Model Selection

1. **Critical Requirement**: For VMEvalKit's reasoning tasks, models MUST accept both:
   - An input image (e.g., maze, chess board, rotation object)
   - A text prompt (e.g., "Show the solution path", "Demonstrate the next move")

2. **Verification Needed**: Many models advertise "image-to-video" but may not support text conditioning. Always verify:
   - Check official documentation for "text-conditioned I2V" or "prompt-guided image animation"
   - Look for API parameters that accept both `image` and `prompt`/`text`
   - Test with sample inputs before full integration

3. **Alternative Approaches**:
   - Some text-to-video models can be adapted using techniques like TI2V-Zero
   - ControlNet-style conditioning can add image guidance to text-to-video models

## 🎯 Model Selection Guide

### Recommended Models for VMEvalKit Tasks

| **Use Case**                          | **Verified Models**                                                                              |
|---------------------------------------|--------------------------------------------------------------------------------------------------|
| **Production-ready APIs**             | Pika 2.0+, Luma Dream Machine, Genmo Mochi                                                    |
| **Enterprise solutions**              | Google Imagen Video, Stability AI Video                                                        |
| **Advanced capabilities**             | Genmo Mochi, Luma Dream Machine                                                                |
| **Best text+image support**           | Pika 2.0+, Luma Dream Machine, Genmo Mochi                                                    |

### Critical Capabilities for Reasoning Tasks

| **Required Feature**              | **Why It's Essential**                                                                |
|----------------------------------|----------------------------------------------------------------------------------------|
| Text prompt input                | Provides instructions like "solve the maze" or "show the next chess move"            |
| Image conditioning               | Preserves the problem visual (maze layout, chess position, etc.)                      |
| Temporal coherence               | Maintains consistency across frames to show logical progression                        |
| Sufficient video length          | Generates enough frames to demonstrate complete solutions                             |

### ⚠️ Model Verification Checklist

Before integrating a model, verify:
- [ ] Accepts both `image` and `text`/`prompt` parameters
- [ ] Documentation explicitly mentions "text-conditioned image-to-video"
- [ ] API/code examples show both inputs being used together
- [ ] Output videos preserve input image content while following text instructions
- [ ] Sufficient video duration for reasoning tasks (minimum 2-4 seconds)

## 🚀 Installation

### Prerequisites
```bash
# Python 3.8 or higher required
python --version

# CUDA 11.8+ for GPU support (recommended)
nvidia-smi
```

### Install from Source
```bash
# Clone the repository
git clone https://github.com/yourusername/VMEvalKit.git
cd VMEvalKit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install VMEvalKit
pip install -e .
```

### Install via pip (coming soon)
```bash
pip install vmevalkit
```

## 📖 Quick Start

### Basic Usage

```python
from vmevalkit import VMEvaluator, TaskLoader, ModelRegistry

# Initialize evaluator
evaluator = VMEvaluator()

# Load a reasoning task (e.g., maze with image + text prompt)
task = TaskLoader.load_task("maze_solving", difficulty="medium")

# Load a closed-source API model that supports text+image
model = ModelRegistry.load_model("pika-2.2", api_key="your-api-key")  # or "luma-dream-machine", "genmo-mochi"

# Run evaluation with both image and text inputs
results = evaluator.evaluate(
    model=model,
    task=task,
    input_image=task.problem_image,  # The maze image
    text_prompt="Navigate through the maze from the green start to the red end point",
    num_samples=5,
    verify_text_image_support=True  # Ensures model accepts both inputs
)

# Print results
print(f"Reasoning Score: {results.reasoning_score:.2f}")
print(f"Video Quality: {results.video_quality:.2f}")
print(f"Solution Correctness: {results.solution_accuracy:.2f}")
```

### Batch Evaluation

```python
import os
from vmevalkit import VMEvaluator

# Evaluate closed-source API models that support text+image→video
api_models = [
    "pika-2.2",          # Pika's text+image video generation
    "luma-dream-machine", # Luma's text+image video generation
    "genmo-mochi",       # Genmo's multimodal video API
    "google-imagen",     # Google's text+image cascade model
    "stability-ai-video" # Stability's text-conditioned animation
]

# Configure API keys
api_keys = {
    "pika-2.2": os.getenv("PIKA_API_KEY"),
    "luma-dream-machine": os.getenv("LUMA_API_KEY"),
    "genmo-mochi": os.getenv("GENMO_API_KEY"),
    "google-imagen": os.getenv("GOOGLE_API_KEY"),
    "stability-ai-video": os.getenv("STABILITY_API_KEY")
}

tasks = ["maze_solving", "mental_rotation", "chess", "ravens_matrices"]

# Run benchmark with API models
benchmark_results = evaluator.run_benchmark(
    models=api_models,
    api_keys=api_keys,
    tasks=tasks,
    output_dir="./results",
    generate_report=True,
    strict_mode=True,  # Fail if model doesn't support text+image
    test_inputs={
        "image": "test_image.png",
        "prompt": "Test prompt for verification"
    }
)
```

## 📊 Evaluation Metrics

### Reasoning Metrics
- **Solution Accuracy**: Correctness of the demonstrated solution
- **Step Validity**: Logical consistency of intermediate steps
- **Completion Rate**: Percentage of successfully completed tasks
- **Planning Efficiency**: Optimality of the solution path

### Video Quality Metrics
- **Temporal Consistency**: Frame-to-frame coherence
- **Visual Clarity**: Sharpness and detail preservation
- **Motion Smoothness**: Natural movement patterns
- **Prompt Adherence**: Alignment with input instructions

## 🏗️ Project Structure

```
VMEvalKit/
├── vmevalkit/
│   ├── core/           # Core evaluation framework
│   ├── models/         # Model interfaces and wrappers
│   ├── tasks/          # Task definitions and datasets
│   ├── metrics/        # Evaluation metrics
│   ├── prompts/        # Prompt templates
│   └── utils/          # Utility functions
├── datasets/           # Task datasets and examples
├── configs/            # Configuration files
├── scripts/            # Utility scripts
├── tests/              # Unit tests
├── docs/               # Documentation
└── examples/           # Example notebooks and scripts
```

## 🔧 Configuration

Create a `config.yaml` file to customize evaluation settings:

```yaml
evaluation:
  batch_size: 4
  num_workers: 8
  seed: 42
  enable_api_batching: true
  rate_limiting:
    max_requests_per_minute: 60
    retry_on_failure: true

models:
  # Closed-source API Models (Supporting Text+Image→Video)
  pika-2.2:
    api_key: "${PIKA_API_KEY}"
    use_pikaframes: true
    duration: 5
    resolution: [1024, 576]
    
  google-veo2:
    api_key: "${GOOGLE_API_KEY}"
    project_id: "${GCP_PROJECT_ID}"
    location: "us-central1"
    quality: "1080p"
    duration: 8
    
  google-imagen-video:
    api_key: "${GOOGLE_API_KEY}"
    project_id: "${GCP_PROJECT_ID}"
    cascade_mode: true
    resolution: [1280, 768]
    
  luma-dream-machine:
    api_key: "${LUMA_API_KEY}"
    enhance_prompt: true
    loop_video: false
    aspect_ratio: "16:9"
    
  stability-ai-video:
    api_key: "${STABILITY_API_KEY}"
    model: "stable-video-v2"
    cfg_scale: 2.5
    motion_bucket_id: 180
    
  genmo-mochi:
    api_key: "${GENMO_API_KEY}"
    hd_mode: true
    duration: 6
    interpolation: "smooth"

tasks:
  maze_solving:
    difficulty_levels: ["easy", "medium", "hard"]
    time_limit: 60
    grid_size: [10, 10]
    require_video_solution: true
    
  mental_rotation:
    object_types: ["3d_shapes", "molecular_structures"]
    rotation_degrees: [90, 180, 270]
    
  chess:
    puzzle_sources: ["lichess", "chess.com"]
    elo_range: [1200, 2000]
    
  ravens_matrices:
    matrix_types: ["2x2", "3x3"]
    pattern_complexity: ["basic", "advanced"]
```

## 📈 Benchmarking

Run comprehensive benchmarks:

```bash
# Run full benchmark suite
python scripts/run_benchmark.py --config configs/full_benchmark.yaml

# Run specific task benchmark
python scripts/run_benchmark.py --task maze_solving --models all

# Generate comparison report
python scripts/generate_report.py --results_dir ./results --output report.html
```

## 🧪 Adding Custom Models

```python
from vmevalkit.models import BaseVideoModel

class MyCustomModel(BaseVideoModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model
    
    def generate(self, prompt, image=None, **kwargs):
        # Implement video generation logic
        return generated_video
    
# Register the model
ModelRegistry.register("my_model", MyCustomModel)
```

## 🎯 Adding Custom Tasks

```python
from vmevalkit.tasks import BaseTask

class MyReasoningTask(BaseTask):
    def __init__(self):
        super().__init__(name="my_task")
    
    def generate_problem(self, difficulty):
        # Create problem instance
        return problem_data
    
    def evaluate_solution(self, video, ground_truth):
        # Evaluate the generated video
        return score

# Register the task
TaskLoader.register_task("my_task", MyReasoningTask)
```

## 📚 Documentation

For detailed documentation, visit our [docs](https://vmevalkit.readthedocs.io) or check the `docs/` directory.

- [API Reference](docs/api_reference.md)
- [Model Integration Guide](docs/model_integration.md)
- [Task Creation Guide](docs/task_creation.md)
- [Evaluation Metrics Details](docs/metrics.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

If you use VMEvalKit in your research, please cite our paper:

```bibtex
@article{vmevalkit2024,
  title={VMEvalKit: A Comprehensive Framework for Evaluating Reasoning in Video Generation Models},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## 🙏 Acknowledgments

We thank the authors of the video generation models and reasoning benchmarks that make this evaluation framework possible.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📮 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/VMEvalKit/issues)

---

<p align="center">
  Made with ❤️ by the VMEvalKit Team
</p>
