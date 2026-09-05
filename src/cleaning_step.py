import pandas as pd

from zenml import step

from src.data_cleaning import clean_sensor_data


@step(
    name="clean_sensor_data",
    enable_artifact_metadata=True,
)
def clean_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Clean the ingested sensor dataset."""

    cleaned_data = clean_sensor_data(data)

    print("Data cleaning completed.")
    print(f"Rows after cleaning: {len(cleaned_data)}")
    print(f"Columns after cleaning: {len(cleaned_data.columns)}")

    print("\nData types after cleaning:")
    print(cleaned_data.dtypes)

    return cleaned_data