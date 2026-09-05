import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "machine_id",
    "ambient_temperature",
    "humidity",
    "temperature",
    "vibration",
    "pressure",
    "rotational_speed",
    "torque",
    "operating_hours",
    "failure",
]


def create_sensor_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create domain-driven features from industrial
    predictive-maintenance sensor data.

    The original DataFrame is not modified.
    """

    data = df.copy()

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ---------------------------------------------------------
    # Timestamp-based features
    # ---------------------------------------------------------

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    data["hour"] = data["timestamp"].dt.hour

    data["day_of_week"] = (
        data["timestamp"].dt.dayofweek
    )

    data["month"] = data["timestamp"].dt.month

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # ---------------------------------------------------------
    # Temperature-related feature
    # ---------------------------------------------------------

    data["temperature_difference"] = (
        data["temperature"]
        - data["ambient_temperature"]
    )

    # ---------------------------------------------------------
    # Mechanical load feature
    # ---------------------------------------------------------

    data["mechanical_load"] = (
        data["rotational_speed"]
        * data["torque"]
    )

    # ---------------------------------------------------------
    # Operating-age feature
    # ---------------------------------------------------------

    data["operating_hours_squared"] = (
        data["operating_hours"] ** 2
    )

    return data