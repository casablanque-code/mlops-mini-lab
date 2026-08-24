"""Разбивает сырые данные на train/test и сохраняет в data/processed."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/dataset.csv")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df["target"]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    print(f"train: {len(train_df)} строк, test: {len(test_df)} строк")


if __name__ == "__main__":
    main()
