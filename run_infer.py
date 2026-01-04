#!/usr/bin/env python3
"""
Simple wrapper for running VMEvalKit inference experiments.
Calls VMEvalKit's generate_videos.py with proper paths.
"""

import subprocess
import sys
from pathlib import Path
import argparse

# Get the VMEvalKit path
VMEVALKIT_DIR = Path(__file__).parent.parent / "VMEvalKit"
GENERATE_SCRIPT = VMEVALKIT_DIR / "examples/generate_videos.py"

def main():
    parser = argparse.ArgumentParser(description="Run VMEvalKit inference")
    parser.add_argument("--model", required=True, nargs="+", help="Models to run")
    parser.add_argument("--questions-dir", default="./data/questions", help="Questions directory")
    parser.add_argument("--output-dir", default="./data/outputs", help="Output directory")
    parser.add_argument("--task-id", nargs="*", help="Specific task IDs to run")
    parser.add_argument("--gpu", type=int, help="Single GPU to use")
    parser.add_argument("--gpus", nargs="+", type=int, help="Multiple GPUs for parallel processing")
    
    args = parser.parse_args()
    
    # Build command for VMEvalKit
    cmd = [
        sys.executable, str(GENERATE_SCRIPT),
        "--model"] + args.model + [
        "--questions-dir", args.questions_dir,
        "--output-dir", args.output_dir
    ]
    
    if args.task_id:
        cmd.extend(["--task-id"] + args.task_id)
    
    if args.gpus:
        cmd.extend(["--gpus"] + [str(g) for g in args.gpus])
    elif args.gpu is not None:
        cmd.extend(["--gpu", str(args.gpu)])
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    # Execute VMEvalKit
    result = subprocess.run(cmd, cwd=VMEVALKIT_DIR)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()