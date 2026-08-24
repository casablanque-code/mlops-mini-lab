"""
Генерирует синтетический датасет для задачи бинарной классификации.
Никаких внешних скачиваний — воспроизводимо и легко для CI/ноутбука.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate(n_samples: int, n_features: int, seed: int) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=1,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=seed,
    )
    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-features", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default="data/raw/dataset.csv"
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    df = generate(args.n_samples, args.n_features, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Сохранено {len(df)} строк в {output_path}")


if __name__ == "__main__":
    main()
