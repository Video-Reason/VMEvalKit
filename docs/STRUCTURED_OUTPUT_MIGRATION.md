# Structured Output Migration Guide

## Overview

VMEvalKit has been enhanced with a **Structured Output System** that creates self-contained folders for each inference. This ensures better organization, reproducibility, and analysis capabilities.

## What Changed?

### Old System (Before)
```
data/outputs/
├── model_name/
│   └── category/
│       └── video.mp4
└── logs/
    └── inference_log.json
```

Videos were scattered across model/category folders with minimal metadata.

### New System (After)
```
output/
└── <model>_<question_id>_<timestamp>/
    ├── video/
    │   └── generated_video.mp4      # The output video
    ├── question/
    │   ├── first_frame.png          # Input image sent to model
    │   ├── final_frame.png          # Reference image (not sent)
    │   ├── prompt.txt               # Text prompt used
    │   └── question_metadata.json   # Complete question data
    └── metadata.json                # Full inference metadata
```

Each inference is now a **complete, self-documenting package**.

## Benefits of the New System

1. **Self-Contained**: All data for an inference in one place
2. **Reproducible**: Input, output, and metadata preserved together
3. **Analyzable**: Structured metadata enables batch analysis
4. **Organized**: Clear folder hierarchy
5. **Complete Context**: Both input and expected output preserved

## How to Use the New System

### Basic Usage

```python
from vmevalkit.runner.inference import InferenceRunner

# Initialize runner with output directory
runner = InferenceRunner(output_dir="output")

# Run inference - automatically creates structured folder
result = runner.run(
    model_name="luma-ray-2",
    image_path="path/to/image.png",
    text_prompt="Your prompt here",
    question_data=question_dict  # Optional: pass full question data
)

print(f"Structured output saved to: {result['inference_dir']}")
```

### With Question Data

When you have question metadata (e.g., from VMEvalKit datasets):

```python
import json

# Load question from dataset
with open("data/questions/vmeval_dataset_v2.json") as f:
    dataset = json.load(f)
    question = dataset["pairs"][0]

# Run inference with question data
result = runner.run(
    model_name="luma-ray-2",
    image_path=question["first_image_path"],
    text_prompt=question["prompt"],
    question_data=question  # Includes final_image_path, metadata, etc.
)
```

The system will automatically:
- Copy both first and final images to the question/ folder
- Save the prompt to a text file
- Preserve all question metadata
- Create comprehensive inference metadata

## Metadata Structure

Each inference folder contains a `metadata.json` with:

```json
{
  "inference": {
    "run_id": "unique_identifier",
    "model": "model_name",
    "timestamp": "ISO_timestamp",
    "status": "success/failed",
    "duration_seconds": 45.2
  },
  "input": {
    "prompt": "text_prompt",
    "image_path": "original_path",
    "question_id": "question_identifier",
    "task_category": "category_name"
  },
  "output": {
    "video_path": "path/to/video",
    "video_url": "optional_url",
    "generation_id": "model_specific_id"
  },
  "paths": {
    "inference_dir": "full/path/to/inference/folder",
    "video_dir": "path/to/video/subfolder",
    "question_dir": "path/to/question/subfolder"
  },
  "question_data": {
    // Full question metadata if provided
  }
}
```

## Migrating Existing Outputs

If you have outputs from the old system, you can:

### Option 1: Keep Old Outputs, Use New for Future
Simply start using the new system for new inferences. Old outputs remain accessible.

### Option 2: Re-run with New System
Re-generate outputs using the new structured system for consistency.

### Option 3: Manual Migration Script
Create a script to reorganize old outputs:

```python
import shutil
import json
from pathlib import Path

def migrate_old_output(old_video_path, question_data, new_output_dir):
    """Migrate old output to new structure."""
    
    # Create new structure
    inference_id = f"migrated_{Path(old_video_path).stem}"
    inference_dir = Path(new_output_dir) / inference_id
    
    # Create directories
    (inference_dir / "video").mkdir(parents=True, exist_ok=True)
    (inference_dir / "question").mkdir(exist_ok=True)
    
    # Copy video
    shutil.copy2(old_video_path, inference_dir / "video" / "video.mp4")
    
    # Copy question data if available
    if question_data:
        if "first_image_path" in question_data:
            shutil.copy2(
                question_data["first_image_path"],
                inference_dir / "question" / "first_frame.png"
            )
        if "final_image_path" in question_data:
            shutil.copy2(
                question_data["final_image_path"],
                inference_dir / "question" / "final_frame.png"
            )
        
        # Save metadata
        metadata = {
            "migration": "from_old_system",
            "original_path": str(old_video_path),
            "question_data": question_data
        }
        with open(inference_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    return inference_dir
```

## S3 Integration

The S3 uploader has been enhanced to handle structured outputs:

```python
from vmevalkit.utils.s3_uploader import S3ImageUploader

uploader = S3ImageUploader()

# Upload entire inference folder
urls = uploader.upload_inference_folder("output/inference_folder_name")

# Returns dictionary of all uploaded files with presigned URLs
print(f"Metadata URL: {urls['metadata.json']}")
```

## Batch Processing

The structured output makes batch analysis easier:

```python
from pathlib import Path
import json

def analyze_all_inferences(output_dir):
    """Analyze all inferences in the output directory."""
    
    results = []
    for inference_dir in Path(output_dir).iterdir():
        if inference_dir.is_dir():
            metadata_file = inference_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    results.append({
                        "inference_id": inference_dir.name,
                        "model": metadata.get("inference", {}).get("model"),
                        "status": metadata.get("inference", {}).get("status"),
                        "duration": metadata.get("inference", {}).get("duration_seconds"),
                        "task_category": metadata.get("input", {}).get("task_category")
                    })
    
    return results

# Analyze all outputs
results = analyze_all_inferences("output")
print(f"Analyzed {len(results)} inferences")
```

## Best Practices

1. **Always pass question_data** when available for complete documentation
2. **Use consistent output_dir** across experiments for easy analysis
3. **Check metadata.json** for debugging failed inferences
4. **Leverage the structure** for automated evaluation pipelines
5. **Archive complete folders** for reproducibility

## Troubleshooting

### Q: Where did my videos go?
A: Videos are now in `output/<inference_id>/video/` instead of scattered across model folders.

### Q: How do I find a specific inference?
A: Look for folders matching the pattern `<model>_<question_id>_<timestamp>` or check the `inference_log.json`.

### Q: Can I use the old system?
A: The new system is now standard. Direct use of `run_inference()` without the runner will still work but won't create the structured output.

### Q: How do I disable structured output?
A: Not recommended, but you can use model wrappers directly instead of InferenceRunner.

## Summary

The new Structured Output System makes VMEvalKit more powerful for:
- **Researchers**: Complete experimental records
- **Developers**: Easy debugging and analysis
- **Teams**: Shareable, self-contained results

Every inference is now a complete story: the problem, the attempt, and the outcome, all in one place.
