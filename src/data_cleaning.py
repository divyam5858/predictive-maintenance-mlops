import pandas as pd


NUMERIC_COLUMNS = [
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


def clean_sensor_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw predictive maintenance dataset.

    The function preserves the original dataframe and
    returns a cleaned copy.
    """

    cleaned_data = data.copy()

    # Remove accidental whitespace from column names.
    cleaned_data.columns = cleaned_data.columns.str.strip()

    # Convert timestamp to a proper datetime representation.
    cleaned_data["timestamp"] = pd.to_datetime(
        cleaned_data["timestamp"],
        errors="coerce",
    )

    # Convert expected numeric columns to numeric types.
    for column in NUMERIC_COLUMNS:
        cleaned_data[column] = pd.to_numeric(
            cleaned_data[column],
            errors="coerce",
        )

    # Remove leading/trailing whitespace from machine IDs
    # when they are represented as strings.
    if cleaned_data["machine_id"].dtype == "object":
        cleaned_data["machine_id"] = (
            cleaned_data["machine_id"]
            .astype(str)
            .str.strip()
        )

    return cleaned_data