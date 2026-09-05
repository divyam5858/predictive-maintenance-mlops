from pathlib import Path

import numpy as np
import pandas as pd


# Locate the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to the raw dataset.
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sensor_data.csv"


def main():
    """Explore the Predictive Maintenance dataset."""

    print("=" * 60)
    print("PREDICTIVE MAINTENANCE DATA EXPLORATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load the dataset
    # ---------------------------------------------------------
    data = pd.read_csv(DATA_PATH)

    print("\n1. DATASET SHAPE")
    print("-" * 40)
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    # ---------------------------------------------------------
    # 2. Display column names
    # ---------------------------------------------------------
    print("\n2. COLUMN NAMES")
    print("-" * 40)

    for column in data.columns:
        print(column)

    # ---------------------------------------------------------
    # 3. Display first records
    # ---------------------------------------------------------
    print("\n3. FIRST FIVE RECORDS")
    print("-" * 40)
    print(data.head())

    # ---------------------------------------------------------
    # 4. Inspect data types
    # ---------------------------------------------------------
    print("\n4. DATA TYPES")
    print("-" * 40)
    print(data.dtypes)

    # ---------------------------------------------------------
    # 5. Dataset information
    # ---------------------------------------------------------
    print("\n5. DATASET INFORMATION")
    print("-" * 40)
    data.info()

    # ---------------------------------------------------------
    # 6. Missing-value analysis
    # ---------------------------------------------------------
    print("\n6. MISSING VALUES")
    print("-" * 40)

    missing_values = data.isnull().sum()

    print(missing_values)

    print(
        f"\nTotal missing values: "
        f"{missing_values.sum()}"
    )

    # ---------------------------------------------------------
    # 7. Duplicate-record analysis
    # ---------------------------------------------------------
    print("\n7. DUPLICATE RECORDS")
    print("-" * 40)

    duplicate_count = data.duplicated().sum()

    print(f"Duplicate records: {duplicate_count}")

    # ---------------------------------------------------------
    # 8. Numerical statistics
    # ---------------------------------------------------------
    print("\n8. NUMERICAL STATISTICS")
    print("-" * 40)

    print(data.describe())

    # ---------------------------------------------------------
    # 9. Machine information
    # ---------------------------------------------------------
    print("\n9. MACHINE INFORMATION")
    print("-" * 40)

    print(
        f"Unique machines: "
        f"{data['machine_id'].nunique()}"
    )

    print(
        f"Machine IDs: "
        f"{data['machine_id'].unique()}"
    )

    # ---------------------------------------------------------
    # 10. Failure distribution
    # ---------------------------------------------------------
    print("\n10. FAILURE DISTRIBUTION")
    print("-" * 40)

    failure_distribution = data["failure"].value_counts()

    print(failure_distribution)

    # ---------------------------------------------------------
    # 11. NumPy-based numerical inspection
    # ---------------------------------------------------------
    print("\n11. NUMPY NUMERICAL INSPECTION")
    print("-" * 40)

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:
        values = data[column].to_numpy()

        print(
            f"{column}: "
            f"min={np.min(values):.4f}, "
            f"max={np.max(values):.4f}, "
            f"mean={np.mean(values):.4f}"
        )

    # ---------------------------------------------------------
    # 12. Timestamp inspection
    # ---------------------------------------------------------
    print("\n12. TIMESTAMP INFORMATION")
    print("-" * 40)

    print(
        f"First timestamp: "
        f"{data['timestamp'].iloc[0]}"
    )

    print(
        f"Last timestamp: "
        f"{data['timestamp'].iloc[-1]}"
    )

    print("\n" + "=" * 60)
    print("DATA EXPLORATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()