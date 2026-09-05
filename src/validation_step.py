import pandas as pd

from zenml import step

from src.data_validation import validate_data
from src.gx_validation import validate_with_great_expectations


@step(
    name="validate_sensor_data",
    enable_artifact_metadata=True,
)
def validate_sensor_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate the ingested sensor dataset using
    Pandas/NumPy checks and Great Expectations.
    """

    print("\nRunning Pandas/NumPy validation...")

    # Run the existing programmatic validation layer.
    validate_data(data)

    print("Pandas/NumPy validation passed.")

    print("\nRunning Great Expectations validation...")

    # Run Great Expectations validation.
    validate_with_great_expectations(data)

    print("\nAll data validation checks passed.")
    print(f"Validated rows: {len(data)}")
    print(f"Validated columns: {len(data.columns)}")

    return data