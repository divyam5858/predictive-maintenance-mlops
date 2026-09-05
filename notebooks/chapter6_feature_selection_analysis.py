from pathlib import Path

import pandas as pd

from src.feature_engineering import create_sensor_features
from src.feature_selection import select_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sensor_data.csv"
)


def main():
    """Analyze the selected model features."""

    data = pd.read_csv(DATA_PATH)

    engineered_data = create_sensor_features(data)

    X, y = select_features(engineered_data)

    print("=" * 60)
    print("PREDICTIVE MAINTENANCE FEATURE SELECTION")
    print("=" * 60)

    print(
        f"\nEngineered dataset shape: "
        f"{engineered_data.shape}"
    )

    print(
        f"Selected feature shape: "
        f"{X.shape}"
    )

    print(
        f"Target shape: "
        f"{y.shape}"
    )

    print("\nSelected features:")
    print("-" * 40)

    for index, feature in enumerate(
        X.columns,
        start=1,
    ):
        print(f"{index}. {feature}")

    print("\nTarget distribution:")
    print("-" * 40)
    print(y.value_counts())

    print("\nFeature selection completed.")


if __name__ == "__main__":
    main()