from pathlib import Path

import pandas as pd

from src.data_validation import (
    detect_outliers_iqr,
    flag_iqr_outliers,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sensor_data.csv"
)


def main():
    """Analyze data-quality characteristics."""

    data = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("PREDICTIVE MAINTENANCE DATA QUALITY ANALYSIS")
    print("=" * 60)

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------
    print("\n1. MISSING VALUE ANALYSIS")
    print("-" * 40)

    missing_values = data.isnull().sum()

    print(missing_values)
    print(
        f"\nTotal missing values: "
        f"{missing_values.sum()}"
    )

    # ---------------------------------------------------------
    # Duplicate records
    # ---------------------------------------------------------
    print("\n2. DUPLICATE RECORD ANALYSIS")
    print("-" * 40)

    duplicate_count = data.duplicated().sum()

    print(
        f"Duplicate records: {duplicate_count}"
    )

    # ---------------------------------------------------------
    # Outlier analysis
    # ---------------------------------------------------------
    print("\n3. OUTLIER ANALYSIS USING IQR")
    print("-" * 40)

    outlier_counts = detect_outliers_iqr(data)

    for column, count in outlier_counts.items():
        print(
            f"{column}: "
            f"{count} potential outliers"
        )


    flagged_data = flag_iqr_outliers(data)

    flagged_count = (
        flagged_data["has_potential_outlier"].sum()
    )

    print("\nOutlier Handling Strategy")
    print("=" * 40)
    print(
        f"Rows containing potential outliers: "
        f"{flagged_count}"
    )
    print(
        "Potential outliers are flagged for "
        "investigation and retained."
    )
    # ---------------------------------------------------------
    # Data-quality summary
    # ---------------------------------------------------------
    print("\n4. DATA QUALITY SUMMARY")
    print("-" * 40)

    print(
        f"Rows: {len(data)}"
    )

    print(
        f"Columns: {len(data.columns)}"
    )

    print(
        f"Missing values: "
        f"{missing_values.sum()}"
    )

    print(
        f"Duplicate records: "
        f"{duplicate_count}"
    )

    print("\n" + "=" * 60)
    print("DATA QUALITY ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()