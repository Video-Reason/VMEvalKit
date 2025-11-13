# HuggingFace Support

This guide explains how to mirror VMEvalKit datasets to the HuggingFace Hub and restore them back locally in the exact same folder structure. This complements the S3-based flow and ensures formats remain consistent with VMEvalKit’s expectations.

## What this enables
- Upload any dataset folder (e.g., `data/questions/`) to a HuggingFace dataset repo, preserving directory layout and filenames.
- Download a complete snapshot from HuggingFace back to a local folder with identical structure.
- Non-interactive CLI and Python API, suitable for automation and CI.

## Install and auth
Dependencies are included in `requirements.txt`:
- `huggingface_hub`
- `datasets`
- `pyarrow`

Authenticate using an environment variable:
```bash
export HF_TOKEN=hf_xxx_your_token_here
```

## CLI usage
The `data/hf_sync.py` utility provides three commands: `upload`, `download`, and `ls`.

### Upload a local folder
```bash
python data/hf_sync.py upload \
  --path data/questions \
  --repo-id your-username/vmevalkit-questions \
  --private \
  --commit-message "Upload questions set" \
  --revision main
```

### Download to a local folder
```bash
python data/hf_sync.py download \
  --repo-id your-username/vmevalkit-questions \
  --target data/questions \
  --revision main
```

### List files in a repo
```bash
python data/hf_sync.py ls \
  --repo-id your-username/vmevalkit-questions \
  --revision main
```

Notes:
- Repos are created automatically if they do not exist (`--private` respected).
- Structure is mirrored exactly, so VMEvalKit can use the data immediately.
- Use separate repos for large artifacts if desired (e.g., `...-outputs`, `...-evaluations`).

## Python API
```python
from pathlib import Path
from data.hf_sync import hf_upload, hf_download, hf_list_files

# Upload folder
hf_upload(
    local_path=Path("data/questions"),
    repo_id="your-username/vmevalkit-questions",
    private=True,
    commit_message="Upload questions",
)

# Download snapshot
out_dir = hf_download(
    repo_id="your-username/vmevalkit-questions",
    target_dir=Path("data/questions"),
)

# List files
files = hf_list_files(repo_id="your-username/vmevalkit-questions")
print(len(files), "files on hub")
```

## Format consistency
The sync preserves VMEvalKit’s layout:
- `data/questions/{domain}_task/{task_id}/` containing `first_frame.png`, `final_frame.png`, `prompt.txt`, and `question_metadata.json`
- `data/outputs/{experiment}/...` and `data/evaluations/{experiment}/...` if you choose to sync those as well

This guarantees seamless use with:
- Runners (`vmevalkit.runner.InferenceRunner`)
- Web dashboard (`web/app.py`)
- Scoring pipelines (`examples/score_videos.py`)

## Optional: Tabular datasets on the Hub
If you prefer a tabular dataset (for metadata-centric workflows), you can still convert `vmeval_dataset.json` to a HuggingFace `datasets` object and push it. See the "HuggingFace Integration" section in `docs/DATA_MANAGEMENT.md` for an example. For complete reproducibility of files, the folder mirroring approach above is recommended.

## Best practices
- Use separate repos for different data categories if they are very large.
- Keep your `HF_TOKEN` in a secure secret store for CI.
- Prefer `download` when collaborating to ensure a consistent layout across machines.

