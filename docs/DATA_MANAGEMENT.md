# VMEvalKit Data Management

This module provides comprehensive data management for VMEvalKit datasets, including version control, S3 synchronization, and dataset organization.

## Overview

VMEvalKit's data management system handles:
- 📁 **Dataset Organization** - Structured storage of questions, inference results, and evaluations
- ☁️ **S3 Synchronization** - Automated backup and sharing via AWS S3
- 🔖 **Version Tracking** - Built-in versioning for reproducibility
- 🤗 **HuggingFace Integration** - Alternative dataset hosting option

## Dataset Structure

VMEvalKit uses a hierarchical structure for organizing all data:

```
data/
├── questions/              # Task datasets
│   ├── vmeval_dataset.json   # Master dataset manifest
│   ├── chess_task/
│   │   └── chess_0000/       # Individual question folder
│   │       ├── first_frame.png    # Initial state
│   │       ├── final_frame.png    # Target state
│   │       ├── prompt.txt         # Text instructions
│   │       └── question_metadata.json  # Task metadata
│   ├── maze_task/
│   ├── raven_task/
│   ├── rotation_task/
│   └── sudoku_task/
├── outputs/                # Inference results
│   └── pilot_experiment/
│       └── <model_name>/
│           └── <task_type>/
│               └── <task_id>/
│                   ├── video/
│                   ├── question/
│                   └── metadata.json
├── evaluations/            # Evaluation results
│   └── pilot_experiment/
│       └── <model_name>/
│           └── <task_type>/
│               └── <task_id>/
│                   ├── human-eval.json
│                   └── GPT4OEvaluator.json
└── data_logging/           # Version tracking
    ├── version_log.json
    └── versions/
```

## S3 Synchronization

### Quick Start

Upload your dataset to S3:

```bash
# Basic upload (uses today's date)
python data/s3_sync.py

# Upload and log version
python data/s3_sync.py --log

# Upload with specific date
python data/s3_sync.py --date 20250115
```

### Configuration

Set up AWS credentials in `.env`:

```bash
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET=vmevalkit
AWS_DEFAULT_REGION=us-east-2
```

### S3 Structure

Data is organized by date on S3:
```
s3://vmevalkit/
├── 20250115/
│   └── data/
│       ├── questions/
│       ├── outputs/
│       └── evaluations/
├── 20250114/
│   └── data/
└── ...
```

### Python API

```python
from data.s3_sync import sync_to_s3

# Upload to S3 with automatic date
s3_uri = sync_to_s3()

# Upload with custom date
s3_uri = sync_to_s3(date_prefix="20250115")

print(f"Data uploaded to: {s3_uri}")
```

## Version Tracking

VMEvalKit includes built-in version tracking for datasets.

### Logging Versions

```bash
# View version history
python data/data_logging/version_tracker.py summary

# Get latest version
python data/data_logging/version_tracker.py latest
```

### Python API

```python
from data.data_logging import log_version, get_latest, print_summary

# Log a new version
log_version(
    version="1.0",
    s3_uri="s3://vmevalkit/20250115/data",
    stats={"size_mb": 180, "files": 1300}
)

# Get latest version info
latest = get_latest()
print(f"Latest: v{latest['version']} at {latest['s3_uri']}")

# Print version summary
print_summary()
```

### Version Log Format

Versions are stored in `data/data_logging/version_log.json`:

```json
{
  "versions": [
    {
      "version": "1.0",
      "date": "20250115",
      "s3_uri": "s3://vmevalkit/20250115/data",
      "size_mb": 180,
      "files": 1300,
      "timestamp": "2025-01-15T10:30:00"
    }
  ]
}
```

## Dataset Creation

### Generate Questions Dataset

Create a new dataset with specified tasks:

```bash
# Generate dataset with 15 questions per domain
python -m vmevalkit.runner.create_dataset --pairs-per-domain 15

# Custom configuration
python -m vmevalkit.runner.create_dataset \
    --pairs-per-domain 20 \
    --domains chess maze sudoku \
    --output-dir data/questions_v2
```

### Dataset Manifest

The master manifest (`vmeval_dataset.json`) tracks all questions:

```json
{
  "version": "1.0",
  "created": "2025-01-15",
  "statistics": {
    "total_questions": 75,
    "domains": {
      "chess": 15,
      "maze": 15,
      "sudoku": 15,
      "rotation": 15,
      "raven": 15
    }
  },
  "questions": [
    {
      "id": "chess_0000",
      "domain": "chess",
      "difficulty": "medium",
      "path": "chess_task/chess_0000"
    }
  ]
}
```

## Data Flow

### 1. Question Creation
```
Task Generator → data/questions/ → S3 Backup
```

### 2. Inference Pipeline
```
Questions → Model Inference → data/outputs/ → S3 Sync
```

### 3. Evaluation Pipeline
```
Outputs → Evaluation (Human/GPT-4O) → data/evaluations/ → S3 Sync
```

## Best Practices

### 1. Regular Backups
```bash
# Daily backup to S3
python data/s3_sync.py --log
```

### 2. Version Before Major Changes
```python
from data.data_logging import log_version

# Before dataset update
log_version("1.1", s3_uri, {"change": "Added 50 new chess puzzles"})
```

### 3. Data Integrity
- Keep `vmeval_dataset.json` in sync with actual files
- Validate dataset structure before experiments
- Use version tracking for reproducibility

### 4. Storage Management
- Archive old experiment outputs to S3
- Keep only active experiments locally
- Use date-based folders for organization

## HuggingFace Integration

Alternative to S3, you can use HuggingFace datasets:

```python
from datasets import Dataset, load_dataset

# Upload to HuggingFace
dataset = load_dataset("json", data_files="data/questions/vmeval_dataset.json")
dataset.push_to_hub("your-username/vmevalkit-questions")

# Download from HuggingFace
dataset = load_dataset("your-username/vmevalkit-questions")
```

## Troubleshooting

### Common Issues

1. **S3 Upload Fails**
   - Check AWS credentials in `.env`
   - Verify bucket permissions
   - Ensure network connectivity

2. **Version Conflict**
   ```python
   # Force overwrite version
   log = load_log()
   log['versions'] = [v for v in log['versions'] if v['version'] != '1.0']
   save_log(log)
   ```

3. **Large Dataset Upload**
   - Use multipart upload for files > 100MB
   - Consider compression before upload
   - Upload in batches

## CLI Commands Summary

| Command | Description |
|---------|-------------|
| `python data/s3_sync.py` | Upload data to S3 |
| `python data/s3_sync.py --log` | Upload and log version |
| `python data/data_logging/version_tracker.py summary` | View version history |
| `python data/data_logging/version_tracker.py latest` | Get latest version |
| `python -m vmevalkit.runner.create_dataset` | Generate new dataset |

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AWS_ACCESS_KEY_ID` | AWS access key | For S3 |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | For S3 |
| `S3_BUCKET` | S3 bucket name (default: vmevalkit) | For S3 |
| `AWS_DEFAULT_REGION` | AWS region (default: us-east-2) | For S3 |

## Related Documentation

- [INFERENCE.md](INFERENCE.md) - How inference results are stored
- [EVALUATION.md](EVALUATION.md) - How evaluation results are organized
- [ADDING_TASKS.md](ADDING_TASKS.md) - Creating new task datasets
