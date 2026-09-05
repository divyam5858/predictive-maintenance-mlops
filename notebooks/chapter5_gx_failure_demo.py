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
    """Demonstrate Great Expectations catching invalid data."""

    # Load the original dataset.
    data = pd.read_csv(DATA_PATH)

    # Convert timestamp to datetime.
    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    print("Original dataset loaded.")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    # Create a copy so the original dataset is never modified.
    bad_data = data.copy()

    # Introduce an invalid failure value.
    bad_data.loc[0, "failure"] = 2

    print("\nIntroduced invalid failure value:")
    print(bad_data.loc[0, "failure"])

    print("\nRunning Great Expectations validation...")

    try:
        validate_with_great_expectations(bad_data)

    except ValueError as error:
        print("\nValidation correctly failed.")
        print(f"Reason: {error}")


if __name__ == "__main__":
    main()