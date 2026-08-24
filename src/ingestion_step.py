from zenml import step
import pandas as pd

from src.data_ingestion import load_sensor_data


@step(
    name="ingest_sensor_data",
    enable_artifact_metadata=True,
)
def ingest_data() -> pd.DataFrame:
    """Load the raw predictive maintenance sensor dataset."""

    df = load_sensor_data()

    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df