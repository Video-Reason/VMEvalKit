#!/bin/bash
# Script to run inference on object_trajectory_task dataset with wan-2.2-ti2v-5b model
# Usage: ./run_object_trajectory.sh [additional arguments]
# Example: ./run_object_trajectory.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUESTIONS_DIR="/home/user1/juyi/video/tmp/VMEvalKit/data/downloads/"
OUTPUT_DIR="${SCRIPT_DIR}/../data/outputs/"
MODEL="wan-2.2-ti2v-5b"
GPULIST=(0 1 2 3 4 5 6 7)

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run inference with the specified questions directory and model
python3 "${SCRIPT_DIR}/run_infer.py" \
    --model "$MODEL" \
    --questions-dir "$QUESTIONS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --gpus "${GPULIST[@]}"
