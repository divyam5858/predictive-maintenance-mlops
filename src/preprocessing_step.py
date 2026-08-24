import pandas as pd
from zenml import step

from src.preprocessing import preprocess_sensor_data


@step(
    name="preprocess_sensor_data",
    enable_artifact_metadata=True,
)
def preprocess_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess sensor data into model features and target."""

    X, y = preprocess_sensor_data(df)

    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Failure distribution:\n{y.value_counts()}")

    return X, y