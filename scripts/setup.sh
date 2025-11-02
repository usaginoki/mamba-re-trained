#!/bin/bash

# Setup script for MAMBA-130M WikiText-103 Training

set -e

echo "======================================================================"
echo "MAMBA-130M WikiText-103 Setup"
echo "======================================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "ERROR: Python 3.11+ is required"
    exit 1
fi

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA available:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: CUDA not detected. Training will be slow on CPU."
fi

echo ""
echo "======================================================================"
echo "Installing dependencies..."
echo "======================================================================"

# Install dependencies
pip install -e .

echo ""
echo "======================================================================"
echo "Installing optimized CUDA kernels (optional, may fail on CPU-only)..."
echo "======================================================================"

# Try to install optimized kernels
pip install causal-conv1d>=1.4.0 mamba-ssm>=2.2.0 --no-build-isolation || {
    echo "WARNING: Failed to install optimized kernels. Training will use fallback implementation."
}

echo ""
echo "======================================================================"
echo "Configuring accelerate for distributed training..."
echo "======================================================================"

# Check if accelerate config exists
if [ -f "$HOME/.cache/huggingface/accelerate/default_config.yaml" ]; then
    echo "Accelerate already configured"
else
    echo "Running accelerate config (you can skip this if only using single GPU)"
    accelerate config || echo "Skipping accelerate config"
fi

echo ""
echo "======================================================================"
echo "Setup completed!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. (Optional) Login to Weights & Biases:"
echo "   wandb login"
echo ""
echo "2. Start training:"
echo "   # Single GPU"
echo "   python -m src.train"
echo ""
echo "   # Multi-GPU (e.g., 4 GPUs)"
echo "   NUM_GPUS=4 bash scripts/train.sh"
echo ""
echo "3. Monitor training:"
echo "   # TensorBoard"
echo "   tensorboard --logdir ./logs"
echo ""
echo "   # Or check W&B dashboard"
echo ""
echo "4. Evaluate trained model:"
echo "   python scripts/evaluate.py --model_path ./outputs/mamba-130m-wikitext103"
echo ""
echo "======================================================================"
echo "For more information, see README.md"
echo "======================================================================"
