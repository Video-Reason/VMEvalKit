#!/bin/bash
##############################################################################
# VMEvalKit - Open-Source Models Inference Runner
# 
# This script runs all 17 open-source models on the complete dataset
# - 1,643 questions across 11 task types
# - 17 open-source models
# - Total: 27,931 video generations (1,643 × 17)
#
# Hardware: 8x NVIDIA H200 (140GB VRAM each)
# Execution: Sequential (one model at a time for stability)
# Resume: Automatically skips completed tasks
##############################################################################

set -e  # Exit on error

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR="data/outputs/pilot_experiment"
LOG_DIR="logs/opensource_inference"
mkdir -p "$LOG_DIR"

# All 17 open-source models
MODELS=(
    # LTX-Video (2 models) - ~16GB VRAM each
    "ltx-video"
    "ltx-video-13b-distilled"
    
    # HunyuanVideo (1 model) - ~24GB VRAM
    "hunyuan-video-i2v"
    
    # VideoCrafter (1 model) - ~16GB VRAM
    "videocrafter2-512"
    
    # DynamiCrafter (3 models) - 12-24GB VRAM
    "dynamicrafter-256"
    "dynamicrafter-512"
    "dynamicrafter-1024"
    
    # Morphic (1 model) - Requires 8 GPUs distributed
    "morphic-frames-to-video"
    
    # Stable Video Diffusion (1 model) - ~20GB VRAM
    "svd"
    
    # WAN (8 models) - 24-48GB VRAM
    "wan"
    "wan-2.1-flf2v-720p"
    "wan-2.2-i2v-a14b"
    "wan-2.1-i2v-480p"
    "wan-2.1-i2v-720p"
    "wan-2.2-ti2v-5b"
    "wan-2.1-vace-14b"
    "wan-2.1-vace-1.3b"
)

# Function to log with timestamp
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Function to check GPU availability
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. GPU required for open-source models."
        exit 1
    fi
    
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    log "Detected ${GPU_COUNT} GPUs"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
}

# Function to run inference for a single model
run_model_inference() {
    local MODEL_NAME=$1
    local GPU_ID=${2:-0}  # Default to GPU 0 if not specified
    
    log "=========================================="
    log "Starting inference for: ${BLUE}${MODEL_NAME}${NC}"
    log "Using GPU: ${GPU_ID}"
    log "=========================================="
    
    local LOG_FILE="${LOG_DIR}/${MODEL_NAME}_$(date '+%Y%m%d_%H%M%S').log"
    
    # Run inference with automatic resume (skips existing outputs)
    CUDA_VISIBLE_DEVICES=${GPU_ID} python examples/generate_videos.py \
        --model "${MODEL_NAME}" \
        --all-tasks \
        2>&1 | tee "${LOG_FILE}"
    
    local EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        log "✅ Completed: ${MODEL_NAME}"
    else
        log_error "❌ Failed: ${MODEL_NAME} (exit code: ${EXIT_CODE})"
        log_warn "Check log file: ${LOG_FILE}"
    fi
    
    return $EXIT_CODE
}

# Main execution
main() {
    log "╔════════════════════════════════════════════════════════════════╗"
    log "║         VMEvalKit Open-Source Models Inference Runner         ║"
    log "╚════════════════════════════════════════════════════════════════╝"
    log ""
    log "Configuration:"
    log "  • Models to run: ${#MODELS[@]}"
    log "  • Questions: 1,643"
    log "  • Total generations: ~27,931"
    log "  • Output directory: ${OUTPUT_DIR}"
    log "  • Log directory: ${LOG_DIR}"
    log ""
    
    # Check GPU availability
    check_gpu
    log ""
    
    # Track statistics
    local TOTAL_MODELS=${#MODELS[@]}
    local COMPLETED=0
    local FAILED=0
    local START_TIME=$(date +%s)
    
    # Sequential execution (one model at a time)
    for i in "${!MODELS[@]}"; do
        local MODEL="${MODELS[$i]}"
        local MODEL_NUM=$((i + 1))
        
        log ""
        log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log "Model ${MODEL_NUM}/${TOTAL_MODELS}: ${MODEL}"
        log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Assign GPU (rotate through available GPUs)
        # For Morphic, we'd need all 8 GPUs, so handle specially
        if [ "$MODEL" = "morphic-frames-to-video" ]; then
            log_warn "Morphic requires all 8 GPUs in distributed mode"
            GPU_ID="0,1,2,3,4,5,6,7"
        else
            # Rotate through GPUs for load balancing
            GPU_ID=$((i % 8))
        fi
        
        # Run inference
        if run_model_inference "$MODEL" "$GPU_ID"; then
            COMPLETED=$((COMPLETED + 1))
        else
            FAILED=$((FAILED + 1))
            log_warn "Continuing with next model..."
        fi
        
        # Show progress
        log ""
        log "Progress: ${COMPLETED} completed, ${FAILED} failed, $((TOTAL_MODELS - MODEL_NUM)) remaining"
        
        # Sleep briefly between models to allow GPU cleanup
        sleep 5
    done
    
    # Final statistics
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    local HOURS=$((DURATION / 3600))
    local MINUTES=$(((DURATION % 3600) / 60))
    local SECONDS=$((DURATION % 60))
    
    log ""
    log "╔════════════════════════════════════════════════════════════════╗"
    log "║                    INFERENCE COMPLETE!                         ║"
    log "╚════════════════════════════════════════════════════════════════╝"
    log ""
    log "Summary:"
    log "  • Total models: ${TOTAL_MODELS}"
    log "  • ✅ Completed: ${COMPLETED}"
    log "  • ❌ Failed: ${FAILED}"
    log "  • ⏱️  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    log ""
    log "Outputs saved to: ${OUTPUT_DIR}"
    log "Logs saved to: ${LOG_DIR}"
    log ""
}

# Handle Ctrl+C gracefully
trap 'log_warn "Interrupted by user. Exiting..."; exit 130' INT

# Run main function
main "$@"

