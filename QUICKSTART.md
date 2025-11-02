# Quick Start Guide

This guide will help you get started with training MAMBA-130M on WikiText-103.

## Setup (5 minutes)

### 1. Install Dependencies

```bash
# Run the setup script
bash scripts/setup.sh

# Or manually:
pip install -e .
```

### 2. (Optional) Configure W&B

```bash
# Login to Weights & Biases for experiment tracking
wandb login
```

## Training

### Option 1: Quick Start (Single GPU)

```bash
# Start training with default settings
python -m src.train
```

This will:
- Initialize MAMBA-130M with random weights
- Load WikiText-103 dataset
- Train for 5 epochs (~4-8 hours on A100)
- Save checkpoints to `./outputs/mamba-130m-wikitext103/`
- Log metrics to TensorBoard and W&B

### Option 2: Multi-GPU Training

```bash
# Train with 4 GPUs
NUM_GPUS=4 bash scripts/train.sh
```

### Option 3: Custom Configuration

1. Edit `configs/mamba_130m.yaml` to customize hyperparameters
2. Run:
```bash
python -m src.train --config configs/mamba_130m.yaml
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./logs
# Open http://localhost:6006
```

### Weights & Biases

Visit: https://wandb.ai/YOUR_USERNAME/mamba-wikitext103

## Evaluation

After training completes:

```bash
# Basic evaluation
python scripts/evaluate.py --model_path ./outputs/mamba-130m-wikitext103

# With text generation samples
python scripts/evaluate.py \
    --model_path ./outputs/mamba-130m-wikitext103 \
    --generate_samples \
    --num_samples 5
```

## Text Generation

### Single Generation

```bash
python scripts/generate.py \
    --model_path ./outputs/mamba-130m-wikitext103 \
    --prompt "The history of artificial intelligence"
```

### Interactive Mode

```bash
python scripts/generate.py \
    --model_path ./outputs/mamba-130m-wikitext103 \
    --interactive
```

Then type prompts interactively!

## Common Commands

```bash
# Resume training from checkpoint
python -m src.train --resume

# Train without W&B
python -m src.train --no_wandb

# Custom output directory
python -m src.train --output_dir ./my_experiment

# Evaluate with baseline comparison
python scripts/evaluate.py \
    --model_path ./outputs/mamba-130m-wikitext103 \
    --compare_baseline
```

## Expected Results

After ~5 epochs of training:
- **Validation Perplexity:** ~25-35
- **Test Perplexity:** ~25-35
- **Training Time (A100):** ~4-8 hours
- **Memory Usage:** ~8-12GB VRAM

## Troubleshooting

### Out of Memory?

Reduce batch size in `configs/mamba_130m.yaml`:
```yaml
training:
  per_device_train_batch_size: 2  # Instead of 4
  gradient_accumulation_steps: 16  # Keep effective batch size
```

### Training Too Slow?

1. Use multiple GPUs: `NUM_GPUS=4 bash scripts/train.sh`
2. Check optimized kernels installed: `pip list | grep mamba`
3. Enable mixed precision (already enabled by default)

### W&B Not Working?

```bash
# Disable W&B
python -m src.train --no_wandb

# Or re-login
wandb login
```

## Next Steps

1. Experiment with different hyperparameters
2. Try different learning rates and batch sizes
3. Fine-tune on domain-specific data
4. Compare with transformer baselines
5. Analyze attention patterns and generation quality

## File Locations

- **Model checkpoints:** `./outputs/mamba-130m-wikitext103/`
- **Training logs:** `./logs/`
- **Configuration:** `configs/mamba_130m.yaml`
- **Metrics:** `./outputs/mamba-130m-wikitext103/all_results.json`

## Getting Help

1. Check `README.md` for detailed documentation
2. Review configuration in `configs/mamba_130m.yaml`
3. Check logs: `tail -f outputs/mamba-130m-wikitext103/trainer_log.txt`
4. Monitor TensorBoard for training curves

Happy training!
