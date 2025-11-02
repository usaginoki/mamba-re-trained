#!/bin/bash

# Training script for MAMBA-130M on WikiText-103
# Supports both single-GPU and multi-GPU distributed training

set -e

# Configuration
NUM_GPUS=${NUM_GPUS:-1}
CONFIG_FILE=${CONFIG_FILE:-"configs/mamba_130m.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/mamba-130m-wikitext103"}
WANDB_PROJECT=${WANDB_PROJECT:-"mamba-wikitext103"}

# Parse command line arguments
RESUME=false
NO_WANDB=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME=true
            shift
            ;;
        --no-wandb)
            NO_WANDB=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build training command
TRAIN_CMD="python -m src.train"

if [ -f "$CONFIG_FILE" ]; then
    TRAIN_CMD="$TRAIN_CMD --config $CONFIG_FILE"
fi

TRAIN_CMD="$TRAIN_CMD --output_dir $OUTPUT_DIR"
TRAIN_CMD="$TRAIN_CMD --wandb_project $WANDB_PROJECT"

if [ "$RESUME" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --resume"
fi

if [ "$NO_WANDB" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --no_wandb"
fi

# Print configuration
echo "======================================================================"
echo "MAMBA-130M WikiText-103 Training"
echo "======================================================================"
echo "Number of GPUs: $NUM_GPUS"
echo "Config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "W&B project: $WANDB_PROJECT"
echo "Resume training: $RESUME"
echo "Disable W&B: $NO_WANDB"
echo "======================================================================"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "Available GPUs:"
    nvidia-smi --list-gpus
    echo "======================================================================"
else
    echo "WARNING: nvidia-smi not found. GPU availability unknown."
fi

# Run training
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU training
    echo "Starting single-GPU training..."
    eval $TRAIN_CMD
else
    # Multi-GPU distributed training with accelerate
    echo "Starting multi-GPU distributed training with $NUM_GPUS GPUs..."

    # Check if accelerate is configured
    if [ ! -f "$HOME/.cache/huggingface/accelerate/default_config.yaml" ]; then
        echo "Accelerate not configured. Running accelerate config..."
        accelerate config
    fi

    # Launch with accelerate
    accelerate launch \
        --num_processes=$NUM_GPUS \
        --mixed_precision=bf16 \
        -m src.train \
        $([ -f "$CONFIG_FILE" ] && echo "--config $CONFIG_FILE") \
        --output_dir $OUTPUT_DIR \
        --wandb_project $WANDB_PROJECT \
        $([ "$RESUME" = true ] && echo "--resume") \
        $([ "$NO_WANDB" = true ] && echo "--no_wandb")
fi

echo "======================================================================"
echo "Training completed!"
echo "======================================================================"
