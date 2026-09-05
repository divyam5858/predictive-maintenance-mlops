from pathlib import Path

import pandas as pd

from src.feature_engineering import create_sensor_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sensor_data.csv"
)


def main():
    """Analyze engineered predictive-maintenance features."""

    data = pd.read_csv(DATA_PATH)

    engineered_data = create_sensor_features(data)

    print("=" * 60)
    print("PREDICTIVE MAINTENANCE FEATURE ENGINEERING")
    print("=" * 60)

    print(
        f"\nOriginal shape: "
        f"{data.shape}"
    )

    print(
        f"Engineered shape: "
        f"{engineered_data.shape}"
    )

    print("\nNew features:")
    print("-" * 40)

    new_features = [
        column
        for column in engineered_data.columns
        if column not in data.columns
    ]

    for feature in new_features:
        print(feature)

    print("\nEngineered feature summary:")
    print("-" * 40)

    print(
        engineered_data[new_features].describe()
    )

    print("\nFeature engineering completed.")


if __name__ == "__main__":
    main()