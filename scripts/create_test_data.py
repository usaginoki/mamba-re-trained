"""Create small test datasets from full parquet files for testing"""

import argparse
from pathlib import Path
import pandas as pd


def create_small_dataset(input_path: str, output_path: str, num_samples: int):
    """Create a small subset of a parquet file

    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        num_samples: Number of samples to extract
    """
    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)

    original_size = len(df)
    print(f"  Original size: {original_size:,} rows")

    # Take first N samples
    df_small = df.head(num_samples)

    print(f"  Creating subset: {len(df_small):,} rows")

    # Save to new parquet file
    df_small.to_parquet(output_path, index=False)

    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    print(f"  Saved to: {output_path} ({file_size:.2f} MB)")

    return len(df_small)


def main():
    parser = argparse.ArgumentParser(description="Create small test datasets from parquet files")
    parser.add_argument(
        "--train_input",
        type=str,
        default="data/train.parquet",
        help="Input training parquet file"
    )
    parser.add_argument(
        "--val_input",
        type=str,
        default="data/val.parquet",
        help="Input validation parquet file"
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="data/train_small.parquet",
        help="Output training parquet file"
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default="data/val_small.parquet",
        help="Output validation parquet file"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=1000,
        help="Number of training samples"
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=100,
        help="Number of validation samples"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Creating Small Test Datasets")
    print("=" * 70)

    # Create train subset
    print("\n1. Creating training subset:")
    train_count = create_small_dataset(
        args.train_input,
        args.train_output,
        args.train_samples
    )

    # Create validation subset
    print("\n2. Creating validation subset:")
    val_count = create_small_dataset(
        args.val_input,
        args.val_output,
        args.val_samples
    )

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Training samples: {train_count:,}")
    print(f"Validation samples: {val_count:,}")
    print(f"\nTrain file: {args.train_output}")
    print(f"Val file: {args.val_output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
