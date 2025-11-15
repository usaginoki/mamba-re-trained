#!/bin/bash

# Production Training Script for MAMBA-130M on Local Parquet Files
# Optimized for 2x RTX 3090 (96GB VRAM)

set -e

echo "======================================================================"
echo "MAMBA-130M Production Training"
echo "======================================================================"
echo "Dataset: 7.46M training examples + 16K validation examples"
echo "Hardware: 2x RTX 3090 (96GB total VRAM)"
echo "Config: Optimized for maximum throughput"
echo "======================================================================"

# Launch multi-GPU training
bash scripts/train_local.sh \
    --num-gpus 2 \
    --config configs/mamba_130m_local_production.yaml \
    --train-parquet data/train.parquet \
    --val-parquet data/val.parquet \
    --output-dir ./outputs/mamba-130m-local-production

echo "======================================================================"
echo "Production training completed!"
echo "======================================================================"
