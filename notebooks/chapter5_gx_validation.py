from pathlib import Path

import pandas as pd

from src.gx_validation import validate_with_great_expectations


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sensor_data.csv"
)


def main():
    """Run Great Expectations validation on sensor data."""

    data = pd.read_csv(DATA_PATH)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    print("Dataset loaded.")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    validate_with_great_expectations(data)


if __name__ == "__main__":
    main()