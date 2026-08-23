import pandas as pd


def preprocess_sensor_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare sensor data for machine learning."""

    data = df.copy()

    # Convert timestamp into datetime format.
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Remove timestamp because the baseline model uses numerical features.
    data = data.drop(columns=["timestamp"])

    # Separate target from input features.
    X = data.drop(columns=["failure"])
    y = data["failure"]

    return X, y