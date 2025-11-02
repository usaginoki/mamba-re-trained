#!/usr/bin/env python3
"""Text Generation Script for MAMBA-130M

Simple script to generate text from a trained MAMBA model.
"""

import argparse
import logging
import sys

import torch
from transformers import AutoTokenizer, MambaForCausalLM

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def generate_text(
    model_path: str,
    prompt: str,
    max_length: int = 200,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    device: str = "cuda"
):
    """Generate text from a prompt

    Args:
        model_path: Path to trained model
        prompt: Input prompt text
        max_length: Maximum generation length
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        num_return_sequences: Number of sequences to generate
        device: Device to use (cuda/cpu)
    """
    # Determine device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    logger.info(f"Loading model from: {model_path}")

    # Load model and tokenizer
    model = MambaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32,
        device_map=device if device == "cuda" else "cpu",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    logger.info(f"Model loaded on {device}")
    logger.info("=" * 70)
    logger.info(f"Prompt: {prompt}")
    logger.info("=" * 70)

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
            do_sample=True if temperature > 0 else False,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and print results
    for i, output in enumerate(outputs, 1):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)

        if num_return_sequences > 1:
            logger.info(f"\nGeneration {i}:")
            logger.info("-" * 70)

        logger.info(generated_text)

    logger.info("=" * 70)


def interactive_mode(model_path: str, **generation_kwargs):
    """Interactive text generation mode

    Args:
        model_path: Path to trained model
        **generation_kwargs: Generation parameters
    """
    logger.info("Entering interactive mode. Type 'quit' or 'exit' to stop.")
    logger.info("=" * 70)

    while True:
        try:
            prompt = input("\nPrompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                logger.info("Exiting interactive mode")
                break

            if not prompt:
                continue

            generate_text(model_path, prompt, **generation_kwargs)

        except KeyboardInterrupt:
            logger.info("\nExiting interactive mode")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate text using trained MAMBA model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for generation (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Maximum generation length (in tokens)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0 = greedy, higher = more random)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for generation"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive mode"
    )

    args = parser.parse_args()

    # Generation kwargs
    generation_kwargs = {
        "max_length": args.max_length,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "num_return_sequences": args.num_sequences,
        "device": args.device,
    }

    # Interactive mode or single generation
    if args.interactive or args.prompt is None:
        interactive_mode(args.model_path, **generation_kwargs)
    else:
        generate_text(args.model_path, args.prompt, **generation_kwargs)


if __name__ == "__main__":
    main()
