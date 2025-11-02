#!/usr/bin/env python3
"""Evaluation script for MAMBA-130M on WikiText-103

This script evaluates a trained MAMBA model on the WikiText-103 test set,
computes perplexity, and generates sample text to verify model quality.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, MambaForCausalLM, set_seed
from datasets import load_dataset
from tqdm import tqdm
import math

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load trained model and tokenizer

    Args:
        model_path: Path to trained model directory
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model from: {model_path}")

    model = MambaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=device if torch.cuda.is_available() else "cpu",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model loaded on {device}")
    logger.info(f"Model dtype: {model.dtype}")

    return model, tokenizer


def compute_perplexity(
    model: MambaForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "test",
    max_length: int = 2048,
    batch_size: int = 1,
    stride: int = 512
) -> float:
    """Compute perplexity on a dataset

    Args:
        model: MAMBA model
        tokenizer: Tokenizer
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        split: Dataset split to evaluate on
        max_length: Maximum sequence length
        batch_size: Batch size for evaluation
        stride: Stride for sliding window

    Returns:
        Perplexity score
    """
    logger.info(f"Computing perplexity on {dataset_name}/{dataset_config} ({split} split)")

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Concatenate all texts
    texts = [text for text in dataset["text"] if text and len(text.strip()) > 0]
    full_text = "\n\n".join(texts)

    # Tokenize
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    seq_len = input_ids.size(1)
    logger.info(f"Total sequence length: {seq_len:,} tokens")

    # Compute loss with sliding window
    nlls = []
    prev_end_loc = 0

    with torch.no_grad():
        for begin_loc in tqdm(range(0, seq_len, stride), desc="Computing perplexity"):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop

            input_ids_chunk = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_chunk.clone()
            target_ids[:, :-trg_len] = -100  # Only compute loss on new tokens

            # Forward pass
            outputs = model(input_ids_chunk, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

    logger.info(f"Perplexity: {ppl.item():.2f}")

    return ppl.item()


def generate_samples(
    model: MambaForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_length: int = 200,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95
):
    """Generate text samples from prompts

    Args:
        model: MAMBA model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
    """
    logger.info("=" * 70)
    logger.info("Generating text samples...")
    logger.info("=" * 70)

    model.eval()

    for i, prompt in enumerate(prompts, 1):
        logger.info(f"\nSample {i}/{len(prompts)}:")
        logger.info(f"Prompt: {prompt}")
        logger.info("-" * 70)

        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info(f"Generated:\n{generated_text}")
        logger.info("=" * 70)


def compare_with_baseline(
    test_ppl: float,
    baseline_model: str = "state-spaces/mamba-130m-hf",
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-103-raw-v1"
):
    """Compare with baseline pretrained model

    Args:
        test_ppl: Perplexity of our trained model
        baseline_model: Baseline model to compare with
        dataset_name: Dataset name
        dataset_config: Dataset config
    """
    logger.info("=" * 70)
    logger.info("Comparing with baseline model...")
    logger.info("=" * 70)

    try:
        # Load baseline model
        baseline_model_obj, baseline_tokenizer = load_model_and_tokenizer(baseline_model)

        # Compute baseline perplexity
        baseline_ppl = compute_perplexity(
            baseline_model_obj,
            baseline_tokenizer,
            dataset_name,
            dataset_config,
            split="test"
        )

        # Compare
        logger.info("\n" + "=" * 70)
        logger.info("Perplexity Comparison:")
        logger.info("=" * 70)
        logger.info(f"Baseline ({baseline_model}): {baseline_ppl:.2f}")
        logger.info(f"Our model: {test_ppl:.2f}")
        logger.info(f"Difference: {test_ppl - baseline_ppl:.2f}")
        logger.info(f"Relative improvement: {(1 - test_ppl/baseline_ppl) * 100:.2f}%")
        logger.info("=" * 70)

    except Exception as e:
        logger.warning(f"Could not compare with baseline: {e}")


def main():
    """Main evaluation entry point"""
    parser = argparse.ArgumentParser(description="Evaluate MAMBA-130M on WikiText-103")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Salesforce/wikitext",
        help="Dataset name"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-103-raw-v1",
        help="Dataset configuration"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride for sliding window perplexity"
    )
    parser.add_argument(
        "--generate_samples",
        action="store_true",
        help="Generate text samples"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        help="Compare with baseline pretrained model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)

    # Compute perplexity
    logger.info("=" * 70)
    logger.info("Evaluating model...")
    logger.info("=" * 70)

    perplexity = compute_perplexity(
        model,
        tokenizer,
        args.dataset,
        args.dataset_config,
        args.split,
        args.max_length,
        args.batch_size,
        args.stride
    )

    # Generate samples if requested
    if args.generate_samples:
        prompts = [
            "The history of artificial intelligence began in the",
            "Machine learning is a subset of",
            "In recent years, deep learning has",
        ][:args.num_samples]

        generate_samples(model, tokenizer, prompts)

    # Compare with baseline if requested
    if args.compare_baseline:
        compare_with_baseline(
            perplexity,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config
        )

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Evaluation Summary:")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Dataset: {args.dataset}/{args.dataset_config} ({args.split})")
    logger.info(f"Perplexity: {perplexity:.2f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
