# Production Training Guide - MAMBA-130M on Local Parquet

## Hardware Setup

- **GPUs**: 2x NVIDIA RTX 3090 (48GB each)
- **Total VRAM**: 96GB
- **System**: Linux with CUDA support

## Dataset

- **Training**: 7,461,630 examples (369 MB)
- **Validation**: 15,974 examples (802 KB)
- **Format**: Parquet files with single `text` column
- **Location**:
  - `data/train.parquet`
  - `data/val.parquet`

## Configuration Highlights

**Batch Size**: Optimized for 96GB VRAM
- Per-device batch size: 32
- Gradient accumulation: 2 steps
- **Effective batch size**: 128 (32 × 2 GPUs × 2 accumulation)

**Training Duration**:
- Epochs: 3
- Total steps: ~175,000 (estimated)
- Evaluation: Every 500 steps
- Checkpointing: Every 1000 steps

**Optimization**:
- Learning rate: 5e-4 with cosine schedule
- Warmup: 2000 steps
- Weight decay: 0.1
- Optimizer: AdamW (β1=0.9, β2=0.95)
- Gradient clipping: 1.0

**Mixed Precision**:
- BF16 enabled (RTX 3090 supported)
- FP32 residual connections for stability

**Memory Optimization**:
- Gradient checkpointing: **Disabled** (not needed with 96GB VRAM)
- Sequence length: Full 2048 tokens
- Data workers: 8 parallel loaders

## Quick Start

### Method 1: Using the Production Script (Recommended)

```bash
bash scripts/train_production.sh
```

This will automatically:
- Use 2 GPUs
- Load the production config
- Start training on full dataset
- Save to `./outputs/mamba-130m-local-production`

### Method 2: Manual Launch

```bash
bash scripts/train_local.sh \
    --num-gpus 2 \
    --config configs/mamba_130m_local_production.yaml \
    --train-parquet data/train.parquet \
    --val-parquet data/val.parquet \
    --output-dir ./outputs/mamba-130m-local-production
```

### Method 3: Resume from Checkpoint

```bash
bash scripts/train_production.sh --resume
```

Or manually:

```bash
bash scripts/train_local.sh \
    --num-gpus 2 \
    --config configs/mamba_130m_local_production.yaml \
    --train-parquet data/train.parquet \
    --val-parquet data/val.parquet \
    --output-dir ./outputs/mamba-130m-local-production \
    --resume
```

## Expected Performance

### Training Speed (Estimated)

With 2x RTX 3090 and batch size 128:
- **~2-3 seconds per step** (estimated)
- **~350-500 steps per hour**
- **~8-12 hours per epoch**
- **~24-36 hours total** for 3 epochs

### Memory Usage (Expected)

- **Per GPU**: ~35-40GB / 48GB
- **Headroom**: ~8-13GB per GPU for safety

### Checkpoints

Checkpoints saved every 1000 steps to:
```
outputs/mamba-130m-local-production/
├── checkpoint-1000/
├── checkpoint-2000/
├── checkpoint-3000/
├── ...
└── checkpoint-final/
```

Last 5 checkpoints are kept (configured by `save_total_limit: 5`).

## Monitoring

### TensorBoard

```bash
tensorboard --logdir logs/mamba-130m-local-production
```

### Weights & Biases

W&B is **enabled** in the production config. Metrics will be logged to:
- Project: `mamba-local-parquet-production`
- Run name: `mamba-130m-7.46M-examples`

To disable W&B:
```bash
bash scripts/train_local.sh \
    --num-gpus 2 \
    --config configs/mamba_130m_local_production.yaml \
    --train-parquet data/train.parquet \
    --val-parquet data/val.parquet \
    --no-wandb
```

### CSV Metrics

Training metrics are also logged to:
```
outputs/mamba-130m-local-production/training_metrics.csv
```

## Optimization Tips

### If You Run Out of Memory

Edit `configs/mamba_130m_local_production.yaml`:

```yaml
training:
  per_device_train_batch_size: 24  # Reduce from 32
  gradient_accumulation_steps: 3   # Increase from 2
  gradient_checkpointing: true     # Enable if needed
```

### To Speed Up Training

1. **Enable fused optimizer** (if available):
   ```yaml
   training:
     optim: "adamw_torch_fused"  # Faster than adamw_torch
   ```

2. **Reduce evaluation frequency**:
   ```yaml
   training:
     eval_steps: 1000  # Increase from 500
   ```

3. **Reduce logging**:
   ```yaml
   training:
     logging_steps: 100  # Increase from 50
   ```

### To Improve Model Quality

1. **Train longer**:
   ```yaml
   training:
     num_train_epochs: 5  # Increase from 3
   ```

2. **Larger batch size**:
   ```yaml
   training:
     per_device_train_batch_size: 40
     gradient_accumulation_steps: 2
   ```

3. **More warmup**:
   ```yaml
   training:
     warmup_steps: 5000  # Increase from 2000
   ```

## Expected Results

After training for 3 epochs, you should see:

- **Training Loss**: ~2.5-3.5 (decreasing over time)
- **Validation Loss**: ~3.0-4.0
- **Validation Perplexity**: ~20-50

These are rough estimates - actual results depend on the data distribution and training dynamics.

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or enable gradient checkpointing (see "If You Run Out of Memory" above).

### Training is Slow

- Check GPU utilization: `nvidia-smi`
- Increase `dataloader_num_workers` if CPU is underutilized
- Consider enabling `adamw_torch_fused` optimizer

### Loss is NaN or Exploding

- Check learning rate (try reducing to 1e-4)
- Verify BF16 is working correctly
- Enable `gradient_checkpointing` for numerical stability

### Checkpoints Taking Too Much Space

Reduce `save_total_limit` in config:
```yaml
training:
  save_total_limit: 3  # Keep only last 3 checkpoints
```

## Files Created

- **Config**: `configs/mamba_130m_local_production.yaml`
- **Launch script**: `scripts/train_production.sh`
- **Data loader**: `src/data_local.py`
- **Training script**: `src/train_local.py`
- **Test data creator**: `scripts/create_test_data.py`

## Next Steps After Training

1. **Evaluate final model**:
   ```bash
   uv run python -m src.evaluate_local \
       --model_path ./outputs/mamba-130m-local-production \
       --test_data data/val.parquet
   ```

2. **Export for inference**:
   The final model is saved in HuggingFace format at:
   ```
   outputs/mamba-130m-local-production/
   ```

3. **Compare with baseline**:
   Compare perplexity with the original `state-spaces/mamba-130m-hf` model.

## Support

For issues or questions:
- Check logs: `outputs/mamba-130m-local-production/`
- TensorBoard: `tensorboard --logdir logs/mamba-130m-local-production`
- CSV metrics: `outputs/mamba-130m-local-production/training_metrics.csv`
