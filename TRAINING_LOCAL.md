# Training MAMBA-130M on Local Parquet Files

This guide explains how to use the new training scripts that load data from local parquet files instead of HuggingFace datasets.

## Overview

The new training implementation consists of:
- **`src/data_local.py`** - Data loading module for local parquet files
- **`src/train_local.py`** - Training script adapted for local data
- **`scripts/train_local.sh`** - Bash launcher script with multi-GPU support

## Quick Start

### Basic Training (Single GPU)

```bash
bash scripts/train_local.sh
```

This will train using the default configuration:
- Training data: `data/train.parquet`
- Validation data: `data/val.parquet`
- Output directory: `./outputs/mamba-130m-local`

### Training with Custom Options

```bash
bash scripts/train_local.sh \
    --train-parquet data/train.parquet \
    --val-parquet data/val.parquet \
    --output-dir ./outputs/my-experiment \
    --config configs/mamba_130m.yaml \
    --no-wandb
```

### Multi-GPU Training

```bash
NUM_GPUS=4 bash scripts/train_local.sh
```

Or:

```bash
bash scripts/train_local.sh --num-gpus 4
```

### Resume from Checkpoint

```bash
bash scripts/train_local.sh --resume
```

## Command Line Options

### Shell Script Options (`train_local.sh`)

- `--train-parquet PATH` - Path to training parquet file (default: `data/train.parquet`)
- `--val-parquet PATH` - Path to validation parquet file (default: `data/val.parquet`)
- `--output-dir PATH` - Output directory for checkpoints and logs (default: `./outputs/mamba-130m-local`)
- `--config PATH` - Path to YAML configuration file (default: `configs/mamba_130m.yaml`)
- `--num-gpus N` - Number of GPUs to use (default: 1)
- `--resume` - Resume training from last checkpoint
- `--no-wandb` - Disable Weights & Biases logging

### Environment Variables

- `NUM_GPUS` - Number of GPUs (alternative to `--num-gpus`)
- `TRAIN_PARQUET` - Training parquet path (alternative to `--train-parquet`)
- `VAL_PARQUET` - Validation parquet path (alternative to `--val-parquet`)
- `OUTPUT_DIR` - Output directory (alternative to `--output-dir`)
- `CONFIG_FILE` - Config file path (alternative to `--config`)
- `WANDB_PROJECT` - W&B project name (default: `mamba-local-parquet`)

## Python Module Usage

You can also run the training script directly with Python:

```bash
# Using uv (recommended for this project)
uv run python -m src.train_local \
    --train_parquet data/train.parquet \
    --val_parquet data/val.parquet \
    --output_dir ./outputs/my-experiment \
    --config configs/mamba_130m.yaml

# Or with standard python
python -m src.train_local \
    --train_parquet data/train.parquet \
    --val_parquet data/val.parquet \
    --output_dir ./outputs/my-experiment
```

### Python Arguments

- `--train_parquet PATH` - Path to training parquet file
- `--val_parquet PATH` - Path to validation parquet file
- `--config PATH` - Configuration file
- `--output_dir PATH` - Output directory
- `--wandb_project NAME` - W&B project name
- `--no_wandb` - Disable W&B logging
- `--resume` - Resume from checkpoint

## Data Format

The parquet files must contain a single column named `text` with string values:

```python
# Expected schema
{
    'text': str  # Raw text content
}
```

Current dataset statistics:
- **Training**: ~7.46M examples (369 MB)
- **Validation**: ~16K examples (802 KB)
- **Test**: Uses validation set (no separate test.parquet)

## Key Differences from Original Training

1. **Data Source**: Loads from local parquet files instead of HuggingFace Hub
2. **Test Set**: Uses validation set for both validation and test evaluation (since no `test.parquet` exists)
3. **File Paths**: Requires specifying parquet file paths
4. **Module Name**: Uses `src.train_local` instead of `src.train`

## Output Files

After training, the output directory will contain:

```
outputs/mamba-130m-local/
├── config.yaml              # Saved configuration
├── training_metrics.csv     # Training metrics log
├── checkpoint-*/            # Model checkpoints
├── all_results.json         # Final metrics
├── train_results.json       # Training results
├── eval_results.json        # Validation results
└── test_results.json        # Test results (same as eval)
```

## Examples

### Minimal Training Run
```bash
bash scripts/train_local.sh --no-wandb
```

### Production Training with Multi-GPU
```bash
bash scripts/train_local.sh \
    --num-gpus 4 \
    --config configs/mamba_130m.yaml \
    --output-dir ./outputs/production-run \
    --wandb-project mamba-production
```

### Debugging (Small Dataset Test)
```bash
# Create a small test dataset first
# Then run with custom paths
bash scripts/train_local.sh \
    --train-parquet data/train_small.parquet \
    --val-parquet data/val_small.parquet \
    --output-dir ./outputs/debug \
    --no-wandb
```

## Troubleshooting

### Import Errors
Make sure you're in the project root directory and using `uv run` or have the virtual environment activated.

### Parquet File Not Found
Verify the file paths:
```bash
ls -lh data/*.parquet
```

### GPU Not Detected
Check GPU availability:
```bash
nvidia-smi
```

### Out of Memory
Reduce batch size in the config file or use gradient accumulation:
```yaml
training:
  per_device_train_batch_size: 4  # Reduce this
  gradient_accumulation_steps: 8  # Increase this
```

## Performance Tips

1. **Use BF16** if your GPU supports it (configured in YAML)
2. **Enable gradient checkpointing** to reduce memory usage
3. **Adjust batch size** based on your GPU memory
4. **Use multiple GPUs** with the `--num-gpus` option
5. **Monitor with W&B** for real-time metrics tracking

## Related Files

- Original training: [src/train.py](src/train.py)
- Original data module: [src/data.py](src/data.py)
- Original shell script: [scripts/train.sh](scripts/train.sh)
- Configuration: [configs/mamba_130m.yaml](configs/mamba_130m.yaml)
