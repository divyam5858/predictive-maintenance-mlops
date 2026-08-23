from zenml import step
import pandas as pd

from src.data_ingestion import load_sensor_data


@step
def ingest_data() -> pd.DataFrame:
    """ZenML step for loading the raw sensor dataset."""

    df = load_sensor_data()

    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df