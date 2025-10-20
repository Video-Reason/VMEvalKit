# VMEvalKit Evaluation Module

This module provides evaluation methods for assessing video generation models' reasoning capabilities.

## Available Evaluators

### 1. Human Evaluator
Interactive web interface for human annotation of generated videos.

**Features:**
- Gradio-based web interface
- Side-by-side display of input and generated video
- Structured evaluation criteria
- Progress tracking
- Export results as JSON

**Usage:**
```bash
# Run human evaluation interface (evaluates entire pilot experiment)
# Automatically creates a public share link for easy access
python examples/run_evaluation.py human
```

### 2. GPT-4O Evaluator
Automatic evaluation using OpenAI's GPT-4O vision model.

**Features:**
- Automatic frame extraction from videos
- Multi-frame analysis
- Task-specific evaluation prompts
- Batch processing of all models
- Detailed scoring and explanations

**Usage:**
```bash
# Evaluate entire pilot experiment (all models, all tasks)
python examples/run_evaluation.py gpt4o
```

**Resume Capability:** Automatically skips already evaluated tasks - safe to re-run.

Note: Set `OPENAI_API_KEY` environment variable before running.

**Requirements:**
- Set `OPENAI_API_KEY` environment variable
- Install dependencies: `opencv-python`, `httpx`

## Evaluation Criteria

Both evaluators use a 1-5 scale for solution correctness:

**Solution Correctness Score**: 
- **1**: Completely wrong solution
- **2**: Mostly incorrect with minor correct elements
- **3**: Partially correct (about half correct)
- **4**: Mostly correct with minor errors
- **5**: Perfect solution

## Output Structure

Evaluations are saved in `data/evaluations/` following this structure:
```
data/evaluations/
├── pilot_experiment/
│   ├── luma-ray-2/
│   │   ├── chess_task/
│   │   │   ├── chess_0000/
│   │   │   │   ├── human-eval.json
│   │   │   │   └── gpt-4o-eval.json
│   │   │   └── ...
│   │   └── ...
│   └── ...
```

Each `*-eval.json` file contains individual evaluation results with metadata. Summary statistics and analysis should be computed separately from these raw evaluation files.

## Custom Evaluators

To create a custom evaluator:

1. Inherit from `BaseEvaluator`
2. Implement `evaluate_single()` method

Example:
```python
from vmevalkit.eval import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    def evaluate_single(self, model_name, task_type, task_id, video_path, question_data):
        # Your evaluation logic here
        return {
            "solution_correctness_score": 5,  # 1-5 scale
            "explanation": "The solution perfectly solves the task"
        }
```

Note: Evaluators focus only on evaluation - analysis and summary statistics should be handled by separate analysis tools.

## API Reference

### BaseEvaluator
Base class providing common evaluation functionality.

**Methods:**
- `evaluate_model(model_name)`: Evaluate all tasks for a model
- `evaluate_all_models()`: Evaluate all models in experiment
- `evaluate_single()`: Abstract method to evaluate one task (implement in subclasses)
- `_save_results()`: Save evaluation results

### HumanEvaluator
Gradio-based interface for human annotation.

**Methods:**
- `launch_interface(share, port)`: Start web interface

### GPT4OEvaluator
Automatic evaluation using GPT-4O.

**Methods:**
- `extract_frames(video_path, num_frames)`: Extract video frames
- `create_evaluation_prompt(task_type)`: Generate task-specific prompts
- `evaluate_single()`: Evaluate one video

## Tips

1. **For Human Evaluation:**
   - The interface automatically creates a public share link
   - Complete evaluations in one session when possible
   - Add detailed comments for edge cases
   - Default annotator name is "Annotator" 

2. **For GPT-4O Evaluation:**
   - Monitor API costs for large experiments
   - Results are deterministic with low temperature (0.1)
   - Check frame extraction quality for debugging

3. **General:**
   - Run evaluations after all inference is complete
   - Compare human and GPT-4O results for validation
   - Usage is simple: `python examples/run_evaluation.py <method>`
