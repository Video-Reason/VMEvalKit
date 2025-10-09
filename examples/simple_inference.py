#!/usr/bin/env python3
"""
Simple example: Veo 3 (Vertex AI) text + image → video for maze reasoning.

Requirements:
- GOOGLE_PROJECT_ID exported (and optionally GOOGLE_LOCATION)
- Auth with `gcloud auth application-default login` or logged-in gcloud

This script generates a video showing a path being drawn through a maze image
according to a reasoning-style prompt, and saves results under ./outputs/.
"""

import os
import asyncio
import json
from pathlib import Path
import sys

# Add parent directory to path so the package resolves when run directly
sys.path.append(str(Path(__file__).parent.parent))

from vmevalkit.models import VeoService


async def main() -> None:
    # Pick a provided maze image (you can change this to any image in data/generated_mazes)
    image_path = "data/generated_mazes/irregular_0000_first.png"

    # Reasoning-oriented prompt: describe how to solve and animate the solution
    prompt = (
        "Solve the maze by tracing a continuous blue path from the green start to the red goal. "
        "Show the path being drawn smoothly over time, with a small leading dot guiding the line. "
        "Do not alter the maze layout or colors; keep the camera fixed and avoid adding text."
    )

    out_path = Path("./outputs/veo_irregular_0000.mp4")

    # Allow overriding project via env or CLI (PROJECT_ID env commonly used)
    project_override = (
        os.getenv("PROJECT_ID")
        or os.getenv("GOOGLE_PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    svc = VeoService(project_id=project_override)
    video_bytes, meta = await svc.generate_video(
        prompt=prompt,
        image_path=image_path,
        duration_seconds=8,
        aspect_ratio="16:9",
        resolution="1080p",
        generate_audio=False,
        sample_count=1,
        download_from_gcs=True,
    )

    # Save metadata for inspection
    meta_path = out_path.with_suffix(".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote operation metadata to {meta_path}")

    if video_bytes:
        saved = await svc.save_video(video_bytes, out_path)
        print(f"✅ Video saved to {saved}")
    else:
        print(
            "No inline bytes returned. If you set a GCS storage URI in the service call, "
            "check your bucket for outputs."
        )


if __name__ == "__main__":
    asyncio.run(main())