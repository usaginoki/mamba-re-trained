"""Debug Training Script - Small subset for quick testing"""

import sys
import torch
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from src.model import initialize_model_from_scratch, create_mamba_config
from src.config import get_default_config

def main():
    print("=" * 70)
    print("DEBUG MODE: Training on small subset")
    print("=" * 70)

    # Load config
    config = get_default_config()

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SMALL subset of data
    print("\n[2/4] Loading SMALL dataset subset...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    # Take only 100 examples for training, 20 for validation
    small_train = dataset["train"].select(range(100))
    small_val = dataset["validation"].select(range(20))

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,  # Shorter sequences for faster debug
            padding=False,
        )

    tokenized_train = small_train.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    tokenized_val = small_val.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    print(f"  Train examples: {len(tokenized_train)}")
    print(f"  Val examples: {len(tokenized_val)}")

    # Initialize model
    print("\n[3/4] Initializing model...")
    model = initialize_model_from_scratch(config.model)
    print(f"  Model initialized with use_cache={model.config.use_cache}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments - minimal for debugging
    training_args = TrainingArguments(
        output_dir="./debug_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,

        # Evaluation
        eval_strategy="steps",
        eval_steps=10,  # Evaluate after just 10 steps

        # Logging
        logging_steps=5,
        logging_first_step=True,

        # Save
        save_strategy="no",

        # Other
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        report_to=[],  # No reporting
        disable_tqdm=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # Trainer
    print("\n[4/4] Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    print("\n" + "=" * 70)
    print("Starting DEBUG training...")
    print("This will train for ~15 steps and evaluate to test the fix")
    print("=" * 70 + "\n")

    try:
        trainer.train()
        print("\n" + "=" * 70)
        print("✓ DEBUG TRAINING SUCCESSFUL!")
        print("The MambaCache error has been fixed.")
        print("=" * 70)
        return 0
    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ DEBUG TRAINING FAILED!")
        print(f"Error: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
