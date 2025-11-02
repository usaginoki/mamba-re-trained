# MAMBA-130M WikiText-103 Training

Training MAMBA-130M language model from scratch on the WikiText-103 dataset.

## Overview

This project implements training for the MAMBA (Structured State Space Model) architecture with 130 million parameters, trained from scratch on WikiText-103. The implementation uses the HuggingFace Transformers library and includes support for:

- Multi-GPU distributed training with DDP
- Mixed precision training (BF16/FP16)
- Experiment tracking with Weights & Biases and TensorBoard
- Gradient checkpointing for memory efficiency
- Comprehensive evaluation and text generation

## Model Architecture

**MAMBA-130M Specifications:**
- Hidden size (d_model): 768
- Number of layers: 24
- Vocabulary size: 50,280
- SSM state size: 16
- Parameters: ~130 million

Based on the architecture from [state-spaces/mamba-130m-hf](https://huggingface.co/state-spaces/mamba-130m-hf).

## Dataset

**WikiText-103** ([Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext))
- 103 million tokens from Wikipedia articles
- Train: 1,801,350 examples
- Validation: 3,760 examples
- Test: 4,358 examples

## Installation

### Prerequisites

- Python 3.11+
- CUDA 11.6+ (for GPU training)
- Linux OS (recommended for optimized kernels)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd project
```

2. Install dependencies:
```bash
pip install -e .
```

Or install with development dependencies:
```bash
pip install -e ".[dev]"
```

### Key Dependencies

- `torch>=2.0.0` - PyTorch
- `transformers>=4.39.0` - HuggingFace Transformers (MAMBA support)
- `datasets>=2.14.0` - HuggingFace Datasets
- `accelerate>=0.24.0` - Distributed training
- `wandb>=0.16.0` - Experiment tracking
- `causal-conv1d>=1.4.0` - Optimized CUDA kernels
- `mamba-ssm>=2.2.0` - MAMBA implementation

## Quick Start

### Single-GPU Training

```bash
# Using default configuration
python -m src.train

# Using custom configuration
python -m src.train --config configs/mamba_130m.yaml
```

### Multi-GPU Training

```bash
# Using the training script
NUM_GPUS=4 bash scripts/train.sh

# Or manually with accelerate
accelerate launch --num_processes=4 --mixed_precision=bf16 -m src.train
```

### Training with Custom Settings

```bash
# Disable W&B logging
python -m src.train --no_wandb

# Resume from checkpoint
python -m src.train --resume

# Custom output directory
python -m src.train --output_dir ./my_outputs
```

## Configuration

### Configuration File

Edit `configs/mamba_130m.yaml` to customize training:

```yaml
# Model architecture
model:
  d_model: 768
  n_layer: 24
  vocab_size: 50280
  # ... more settings

# Training hyperparameters
training:
  num_train_epochs: 5
  per_device_train_batch_size: 4
  learning_rate: 5.0e-4
  # ... more settings
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_epochs` | 5 | Number of training epochs |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `gradient_accumulation_steps` | 8 | Effective batch size = 32 |
| `learning_rate` | 5e-4 | Peak learning rate |
| `max_seq_length` | 2048 | Maximum sequence length |
| `bf16` | true | Use BF16 mixed precision |
| `gradient_checkpointing` | true | Enable gradient checkpointing |

## Training Details

### Hardware Requirements

**Minimum:**
- 1x GPU with 12GB+ VRAM (RTX 3090, RTX 4090)
- 32GB+ system RAM

**Recommended:**
- 1x A100 (40GB) or V100 (32GB)
- 64GB+ system RAM

### Estimated Training Time

**Single GPU (A100):**
- ~4-8 hours for 5 epochs on WikiText-103
- ~103M tokens processed

**Multi-GPU (4x A100):**
- ~1-2 hours for 5 epochs

### Memory Usage

- Single GPU training: ~8-12GB VRAM
- With gradient checkpointing: ~6-8GB VRAM
- Mixed precision (BF16) reduces memory by ~40%

## Evaluation

### Evaluate Trained Model

```bash
# Basic evaluation
python scripts/evaluate.py --model_path ./outputs/mamba-130m-wikitext103

# With sample generation
python scripts/evaluate.py \
  --model_path ./outputs/mamba-130m-wikitext103 \
  --generate_samples \
  --num_samples 5

# Compare with baseline
python scripts/evaluate.py \
  --model_path ./outputs/mamba-130m-wikitext103 \
  --compare_baseline
```

### Expected Results

**Target Perplexity:**
- Validation: ~25-35 (depending on training)
- Test: ~25-35

The baseline pretrained MAMBA-130M achieves similar perplexity on WikiText-103.

## Project Structure

```
project/
├── configs/
│   └── mamba_130m.yaml          # Model and training configuration
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration dataclasses
│   ├── data.py                  # Dataset loading and preprocessing
│   ├── model.py                 # Model initialization
│   └── train.py                 # Main training script
├── scripts/
│   ├── train.sh                 # Training launch script
│   └── evaluate.py              # Evaluation script
├── outputs/                     # Training outputs (checkpoints, logs)
├── logs/                        # TensorBoard logs
├── pyproject.toml               # Project dependencies
└── README.md                    # This file
```

## Distributed Training

### Using Accelerate

1. Configure accelerate (first time only):
```bash
accelerate config
```

2. Launch training:
```bash
accelerate launch -m src.train --config configs/mamba_130m.yaml
```

### Using the Training Script

```bash
# 4 GPUs
NUM_GPUS=4 bash scripts/train.sh

# With custom config
NUM_GPUS=4 CONFIG_FILE=configs/custom.yaml bash scripts/train.sh

# Resume training
NUM_GPUS=4 bash scripts/train.sh --resume
```

## Experiment Tracking

### Weights & Biases

1. Login to W&B:
```bash
wandb login
```

2. Training will automatically log to W&B:
   - Loss curves
   - Learning rate schedule
   - Perplexity metrics
   - Model checkpoints
   - System metrics (GPU, memory)

3. View experiments at: https://wandb.ai/YOUR_USERNAME/mamba-wikitext103

### TensorBoard

```bash
# View logs
tensorboard --logdir ./logs

# Or from outputs
tensorboard --logdir ./outputs/mamba-130m-wikitext103/logs
```

## Advanced Usage

### Resume Training from Checkpoint

```bash
python -m src.train --resume --config configs/mamba_130m.yaml
```

### Custom Data Preprocessing

Edit `src/data.py` to customize data loading:

```python
# Example: Enable text grouping for more efficient training
tokenized_dataset, tokenizer = prepare_dataset(
    config.data,
    use_grouped_texts=True  # Group texts into fixed-size blocks
)
```

### Fine-tuning Pretrained Model

Edit `configs/mamba_130m.yaml`:

```yaml
# Set to false to load pretrained weights
train_from_scratch: false
```

Then:
```bash
python -m src.train --config configs/mamba_130m.yaml
```

## Troubleshooting

### CUDA Out of Memory

1. Reduce batch size:
   ```yaml
   per_device_train_batch_size: 2  # Instead of 4
   gradient_accumulation_steps: 16  # Maintain effective batch size
   ```

2. Enable gradient checkpointing (already enabled by default)

3. Reduce sequence length:
   ```yaml
   max_seq_length: 1024  # Instead of 2048
   ```

### Slow Training

1. Install optimized kernels:
   ```bash
   pip install causal-conv1d>=1.4.0 mamba-ssm>=2.2.0 --no-build-isolation
   ```

2. Enable mixed precision (already enabled):
   ```yaml
   bf16: true  # Or fp16: true
   ```

3. Use multiple GPUs for distributed training

### W&B Authentication Issues

```bash
# Re-login
wandb login

# Or disable W&B
python -m src.train --no_wandb
```

## Citation

If you use this code, please cite the original MAMBA paper:

```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

## License

This project is licensed under the same license as the original MAMBA implementation.

## Acknowledgements

- [state-spaces/mamba](https://github.com/state-spaces/mamba) - Original MAMBA implementation
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - MAMBA integration
- [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) - Dataset

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the configuration in `configs/mamba_130m.yaml`
3. Check logs in `outputs/` or TensorBoard
4. Open an issue on GitHub
