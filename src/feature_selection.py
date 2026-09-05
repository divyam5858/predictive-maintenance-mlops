import pandas as pd


TARGET_COLUMN = "failure"

NON_FEATURE_COLUMNS = [
    "timestamp",
]


SELECTED_FEATURES = [
    "machine_id",
    "ambient_temperature",
    "humidity",
    "temperature",
    "vibration",
    "pressure",
    "rotational_speed",
    "torque",
    "operating_hours",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "temperature_difference",
    "mechanical_load",
    "operating_hours_squared",
]


def select_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select model features and separate the target.

    The timestamp column is excluded because its useful
    temporal information has already been transformed into
    explicit time-based features.

    Returns:
        X: Selected model features.
        y: Target variable.
    """

    data = df.copy()

    required_columns = (
        SELECTED_FEATURES
        + NON_FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    X = data[SELECTED_FEATURES].copy()

    y = data[TARGET_COLUMN].copy()

    return X, y