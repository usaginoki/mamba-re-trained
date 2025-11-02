"""Main Training Script for MAMBA-130M on WikiText-103"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from .config import ExperimentConfig, get_default_config, load_config_from_yaml
from .data import prepare_dataset, get_data_collator, compute_dataset_metrics
from .model import (
    get_model_and_tokenizer,
    print_model_architecture,
    verify_model_config,
    count_parameters,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def setup_wandb(config: ExperimentConfig):
    """Setup Weights & Biases logging

    Args:
        config: Experiment configuration
    """
    if not config.use_wandb:
        logger.info("W&B logging disabled")
        return

    try:
        import wandb

        # Initialize W&B
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.name,
            tags=config.wandb.tags,
            notes=config.wandb.notes,
            config={
                "model": config.model.__dict__,
                "data": config.data.__dict__,
                "training": config.training.__dict__,
            }
        )

        logger.info(f"W&B initialized: {wandb.run.name}")

    except ImportError:
        logger.warning("wandb not installed. Skipping W&B logging.")
        config.use_wandb = False
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}")
        config.use_wandb = False


def compute_metrics(eval_preds):
    """Compute evaluation metrics (perplexity)

    Args:
        eval_preds: Evaluation predictions from trainer

    Returns:
        Dictionary of metrics
    """
    import numpy as np

    predictions, labels = eval_preds

    # For causal LM, predictions are logits
    # Compute perplexity from loss (calculated by trainer)
    # This is a placeholder - actual loss is computed by trainer

    return {}  # Trainer computes loss automatically


def create_training_arguments(config: ExperimentConfig) -> TrainingArguments:
    """Create TrainingArguments from config

    Args:
        config: Experiment configuration

    Returns:
        TrainingArguments instance
    """
    training_config = config.training

    # Determine which precision to use
    if torch.cuda.is_bf16_supported():
        bf16 = training_config.bf16
        fp16 = False if bf16 else training_config.fp16
        logger.info(f"Using BF16: {bf16}, FP16: {fp16}")
    else:
        bf16 = False
        fp16 = training_config.fp16
        logger.info(f"BF16 not supported. Using FP16: {fp16}")

    args = TrainingArguments(
        # Output
        output_dir=training_config.output_dir,
        run_name=training_config.run_name,
        logging_dir=training_config.logging_dir,

        # Training duration
        num_train_epochs=training_config.num_train_epochs,
        max_steps=training_config.max_steps,

        # Batch size
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,

        # Optimization
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        adam_beta1=training_config.adam_beta1,
        adam_beta2=training_config.adam_beta2,
        adam_epsilon=training_config.adam_epsilon,
        max_grad_norm=training_config.max_grad_norm,
        optim=training_config.optim,

        # Learning rate schedule
        lr_scheduler_type=training_config.lr_scheduler_type,
        warmup_steps=training_config.warmup_steps,
        warmup_ratio=training_config.warmup_ratio,

        # Mixed precision
        fp16=fp16,
        bf16=bf16,
        fp16_full_eval=training_config.fp16_full_eval,

        # Memory optimization
        gradient_checkpointing=training_config.gradient_checkpointing,

        # Evaluation
        eval_strategy=training_config.eval_strategy,
        eval_steps=training_config.eval_steps,
        eval_on_start=False,

        # Saving
        save_strategy=training_config.save_strategy,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        load_best_model_at_end=training_config.load_best_model_at_end,
        metric_for_best_model=training_config.metric_for_best_model,
        greater_is_better=training_config.greater_is_better,

        # Logging
        logging_steps=training_config.logging_steps,
        logging_first_step=training_config.logging_first_step,
        report_to=training_config.report_to if config.use_wandb else ["tensorboard"],

        # Distributed training
        ddp_backend=training_config.ddp_backend,
        ddp_find_unused_parameters=training_config.ddp_find_unused_parameters,

        # Reproducibility
        seed=training_config.seed,
        data_seed=training_config.data_seed,

        # Other
        remove_unused_columns=training_config.remove_unused_columns,
        label_names=training_config.label_names,
        include_inputs_for_metrics=training_config.include_inputs_for_metrics,

        # Misc
        dataloader_num_workers=config.data.dataloader_num_workers,
        dataloader_pin_memory=True,
        disable_tqdm=False,
    )

    return args


def train(config: ExperimentConfig, resume_from_checkpoint: bool = False):
    """Main training function

    Args:
        config: Experiment configuration
        resume_from_checkpoint: Whether to resume from checkpoint
    """
    # Set seed for reproducibility
    set_seed(config.training.seed)

    # Setup W&B
    setup_wandb(config)

    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)

    # Save configuration
    from .config import save_config_to_yaml
    config_path = Path(config.training.output_dir) / "config.yaml"
    save_config_to_yaml(config, str(config_path))
    logger.info(f"Configuration saved to: {config_path}")

    # Load data
    logger.info("=" * 70)
    logger.info("Loading and preprocessing dataset...")
    logger.info("=" * 70)

    tokenized_dataset, tokenizer = prepare_dataset(
        config.data,
        use_grouped_texts=False  # Set to True for more efficient packing
    )

    # Compute and log dataset metrics
    dataset_metrics = compute_dataset_metrics(tokenized_dataset)
    logger.info(f"\nDataset metrics: {dataset_metrics}")

    # Get data collator
    data_collator = get_data_collator(tokenizer, mlm=False)

    # Initialize model
    logger.info("=" * 70)
    logger.info("Initializing model...")
    logger.info("=" * 70)

    model, _ = get_model_and_tokenizer(
        model_config=config.model,
        tokenizer_name=config.data.tokenizer_name,
        from_scratch=config.train_from_scratch,
        device=None  # Let Trainer handle device placement
    )

    # Print model architecture
    print_model_architecture(model)

    # Verify configuration
    if config.train_from_scratch:
        verify_model_config(model, config.model)

    # Enable gradient checkpointing if specified
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Create training arguments
    training_args = create_training_arguments(config)

    # Check for existing checkpoints
    last_checkpoint = None
    if resume_from_checkpoint:
        last_checkpoint = get_last_checkpoint(config.training.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
        else:
            logger.info("No checkpoint found, starting from scratch")

    # Setup callbacks
    callbacks = []

    # Early stopping
    if config.training.early_stopping_patience > 0:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=config.training.early_stopping_patience,
            early_stopping_threshold=config.training.early_stopping_threshold
        )
        callbacks.append(early_stopping)
        logger.info(f"Early stopping enabled (patience={config.training.early_stopping_patience})")

    # Initialize Trainer
    logger.info("=" * 70)
    logger.info("Initializing Trainer...")
    logger.info("=" * 70)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # Log training info
    logger.info(f"Number of training examples: {len(tokenized_dataset['train']):,}")
    logger.info(f"Number of validation examples: {len(tokenized_dataset['validation']):,}")
    logger.info(f"Number of epochs: {config.training.num_train_epochs}")
    logger.info(f"Batch size per device: {config.training.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
    logger.info(f"Total batch size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps * max(1, torch.cuda.device_count())}")
    logger.info(f"Learning rate: {config.training.learning_rate}")

    # Train
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save final model
    logger.info("=" * 70)
    logger.info("Training completed! Saving model...")
    logger.info("=" * 70)

    trainer.save_model()
    trainer.save_state()

    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Evaluate on validation set
    logger.info("=" * 70)
    logger.info("Evaluating on validation set...")
    logger.info("=" * 70)

    eval_metrics = trainer.evaluate()

    # Compute perplexity
    try:
        perplexity = torch.exp(torch.tensor(eval_metrics["eval_loss"]))
        eval_metrics["perplexity"] = perplexity.item()
        logger.info(f"Validation Perplexity: {perplexity.item():.2f}")
    except:
        logger.warning("Could not compute perplexity")

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Evaluate on test set
    logger.info("=" * 70)
    logger.info("Evaluating on test set...")
    logger.info("=" * 70)

    test_metrics = trainer.evaluate(tokenized_dataset["test"], metric_key_prefix="test")

    try:
        test_perplexity = torch.exp(torch.tensor(test_metrics["test_loss"]))
        test_metrics["test_perplexity"] = test_perplexity.item()
        logger.info(f"Test Perplexity: {test_perplexity.item():.2f}")
    except:
        logger.warning("Could not compute test perplexity")

    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    logger.info("=" * 70)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 70)

    # Finish W&B run
    if config.use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train MAMBA-130M on WikiText-103")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Override W&B project name"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config_from_yaml(args.config)
    else:
        logger.info("Using default configuration")
        config = get_default_config()

    # Apply overrides
    if args.output_dir:
        config.training.output_dir = args.output_dir

    if args.wandb_project:
        config.wandb.project = args.wandb_project

    if args.no_wandb:
        config.use_wandb = False

    # Log configuration
    logger.info("=" * 70)
    logger.info("Experiment Configuration:")
    logger.info("=" * 70)
    logger.info(f"Model: MAMBA-{config.model.d_model}M")
    logger.info(f"Dataset: {config.data.dataset_name}/{config.data.dataset_config}")
    logger.info(f"Output directory: {config.training.output_dir}")
    logger.info(f"Training from scratch: {config.train_from_scratch}")
    logger.info(f"Use W&B: {config.use_wandb}")
    logger.info("=" * 70)

    # Start training
    train(config, resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
