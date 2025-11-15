"""Data Loading and Preprocessing for Local Parquet Files"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from .config import DataConfig

logger = logging.getLogger(__name__)


def load_parquet_dataset(
    train_path: str,
    val_path: str,
    config: DataConfig,
    tokenizer: Optional[AutoTokenizer] = None
) -> DatasetDict:
    """Load and preprocess dataset from local parquet files

    Args:
        train_path: Path to training parquet file
        val_path: Path to validation parquet file
        config: Data configuration
        tokenizer: Tokenizer for preprocessing (if None, will load from config)

    Returns:
        DatasetDict with train, validation, and test splits (test = validation)
    """
    logger.info(f"Loading dataset from local parquet files:")
    logger.info(f"  Train: {train_path}")
    logger.info(f"  Validation: {val_path}")

    # Verify files exist
    if not Path(train_path).exists():
        raise FileNotFoundError(f"Training parquet file not found: {train_path}")
    if not Path(val_path).exists():
        raise FileNotFoundError(f"Validation parquet file not found: {val_path}")

    # Load dataset from parquet files using pandas (avoids cache issues)
    logger.info("Reading parquet files with pandas...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    logger.info(f"  Train: {len(train_df):,} rows")
    logger.info(f"  Val: {len(val_df):,} rows")

    # Convert to HuggingFace datasets
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df, preserve_index=False),
        'validation': Dataset.from_pandas(val_df, preserve_index=False),
    })

    logger.info(f"Dataset loaded: {dataset}")

    # Load tokenizer if not provided
    if tokenizer is None:
        logger.info(f"Loading tokenizer: {config.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(
            examples,
            tokenizer,
            max_length=config.max_seq_length
        ),
        batched=True,
        num_proc=config.preprocessing_num_workers,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=not config.overwrite_cache,
        desc="Tokenizing dataset"
    )

    # Use validation set for test set as well (since no test.parquet exists)
    logger.info("Using validation set for test evaluation (no separate test file)")
    tokenized_dataset['test'] = tokenized_dataset['validation']

    logger.info(f"Tokenized dataset: {tokenized_dataset}")

    return tokenized_dataset


def tokenize_function(
    examples: Dict,
    tokenizer: AutoTokenizer,
    max_length: int = 2048
) -> Dict:
    """Tokenize text examples

    Args:
        examples: Batch of examples with 'text' field
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Dictionary with tokenized inputs
    """
    # Filter out empty texts
    texts = [text for text in examples["text"] if text and len(text.strip()) > 0]

    if not texts:
        return {"input_ids": [], "attention_mask": []}

    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # Dynamic padding in collator
        return_attention_mask=True
    )

    return tokenized


def group_texts(examples: Dict, block_size: int = 2048) -> Dict:
    """Group texts into fixed-size blocks for more efficient training

    This concatenates all texts and splits them into chunks of block_size.
    Useful for language modeling on long documents.

    Args:
        examples: Batch of tokenized examples
        block_size: Size of each block

    Returns:
        Dictionary with grouped texts
    """
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # Drop last chunk if incomplete
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # Split by chunks
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    # Create labels (copy of input_ids for causal LM)
    result["labels"] = result["input_ids"].copy()

    return result


def get_data_collator(tokenizer: AutoTokenizer, mlm: bool = False):
    """Get data collator for language modeling

    Args:
        tokenizer: Tokenizer to use
        mlm: Whether to use masked language modeling (False for causal LM)

    Returns:
        DataCollatorForLanguageModeling instance
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
        pad_to_multiple_of=8  # For better GPU efficiency
    )


def prepare_dataset_from_parquet(
    train_path: str,
    val_path: str,
    config: DataConfig,
    use_grouped_texts: bool = False
) -> tuple[DatasetDict, AutoTokenizer]:
    """Prepare dataset from local parquet files for training

    Args:
        train_path: Path to training parquet file
        val_path: Path to validation parquet file
        config: Data configuration
        use_grouped_texts: Whether to group texts into fixed-size blocks

    Returns:
        Tuple of (tokenized_dataset, tokenizer)
    """
    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Load and tokenize dataset from parquet
    tokenized_dataset = load_parquet_dataset(
        train_path, val_path, config, tokenizer
    )

    # Optionally group texts
    if use_grouped_texts:
        logger.info("Grouping texts into fixed-size blocks...")
        tokenized_dataset = tokenized_dataset.map(
            lambda examples: group_texts(examples, config.max_seq_length),
            batched=True,
            num_proc=config.preprocessing_num_workers,
            desc="Grouping texts"
        )

    # Log dataset statistics
    log_dataset_statistics(tokenized_dataset, tokenizer)

    return tokenized_dataset, tokenizer


def log_dataset_statistics(dataset: DatasetDict, tokenizer: AutoTokenizer):
    """Log dataset statistics

    Args:
        dataset: Dataset to analyze
        tokenizer: Tokenizer used
    """
    logger.info("=" * 50)
    logger.info("Dataset Statistics:")
    logger.info("=" * 50)

    for split_name, split_data in dataset.items():
        num_examples = len(split_data)
        logger.info(f"\n{split_name.upper()} split:")
        logger.info(f"  Number of examples: {num_examples:,}")

        if num_examples > 0 and "input_ids" in split_data.column_names:
            # Calculate average sequence length
            first_batch = split_data.select(range(min(100, num_examples)))
            avg_length = sum(len(x) for x in first_batch["input_ids"]) / len(first_batch)
            logger.info(f"  Average sequence length: {avg_length:.1f} tokens")

            # Show example
            if num_examples > 0:
                example = split_data[0]
                logger.info(f"  Example input_ids shape: {len(example['input_ids'])}")
                logger.info(f"  Example text (first 200 chars):")
                decoded = tokenizer.decode(example["input_ids"][:100])
                logger.info(f"    {decoded[:200]}...")

    logger.info("=" * 50)


def compute_dataset_metrics(dataset: DatasetDict) -> Dict:
    """Compute detailed dataset metrics

    Args:
        dataset: Dataset to analyze

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    for split_name, split_data in dataset.items():
        split_metrics = {
            "num_examples": len(split_data),
        }

        if "input_ids" in split_data.column_names and len(split_data) > 0:
            # Compute sequence length statistics
            lengths = [len(x) for x in split_data["input_ids"]]
            split_metrics.update({
                "avg_seq_length": sum(lengths) / len(lengths),
                "min_seq_length": min(lengths),
                "max_seq_length": max(lengths),
                "total_tokens": sum(lengths)
            })

        metrics[split_name] = split_metrics

    return metrics
