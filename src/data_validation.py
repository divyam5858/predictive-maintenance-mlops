import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
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


SENSOR_COLUMNS = [
    "ambient_temperature",
    "humidity",
    "temperature",
    "vibration",
    "pressure",
    "rotational_speed",
    "torque",
    "operating_hours",
]

EXPECTED_DTYPES = {
    "timestamp": "datetime64[ns]",
    "machine_id": "int64",
    "ambient_temperature": "float64",
    "humidity": "float64",
    "temperature": "float64",
    "vibration": "float64",
    "pressure": "float64",
    "rotational_speed": "float64",
    "torque": "float64",
    "operating_hours": "float64",
    "failure": "int64",
}

def check_required_columns(data: pd.DataFrame) -> None:
    """Check that all required columns are present."""

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

def check_data_types(data: pd.DataFrame) -> None:
    """Check that columns have the expected data types."""

    incorrect_types = {}

    for column, expected_dtype in EXPECTED_DTYPES.items():
        actual_dtype = str(data[column].dtype)

        if actual_dtype != expected_dtype:
            incorrect_types[column] = {
                "expected": expected_dtype,
                "actual": actual_dtype,
            }

    if incorrect_types:
        raise ValueError(
            f"Data type mismatch detected: "
            f"{incorrect_types}"
        )

def check_timestamp(data: pd.DataFrame) -> None:
    """Check that timestamp values are valid datetimes."""

    if not pd.api.types.is_datetime64_any_dtype(
        data["timestamp"]
    ):
        raise ValueError(
            "Timestamp column must contain datetime values."
        )

    if data["timestamp"].isnull().any():
        raise ValueError(
            "Invalid or missing timestamp values detected."
        )


def check_missing_values(data: pd.DataFrame) -> None:
    """Check for missing values in the dataset."""

    missing_values = data.isnull().sum()

    columns_with_missing_values = missing_values[
        missing_values > 0
    ]

    if not columns_with_missing_values.empty:
        raise ValueError(
            "Missing values detected:\n"
            f"{columns_with_missing_values}"
        )


def check_duplicate_records(data: pd.DataFrame) -> None:
    """Check for duplicate records."""

    duplicate_count = data.duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate records."
        )


def check_numeric_values(data: pd.DataFrame) -> None:
    """Check that numeric values are finite."""

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns

    numeric_values = data[numeric_columns].to_numpy()

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Non-finite numeric values detected."
        )


def check_failure_values(data: pd.DataFrame) -> None:
    """Check that the failure target contains valid values."""

    valid_values = {0, 1}

    actual_values = set(
        data["failure"].dropna().unique()
    )

    invalid_values = actual_values - valid_values

    if invalid_values:
        raise ValueError(
            f"Invalid failure values detected: "
            f"{invalid_values}"
        )
    
def check_machine_ids(data: pd.DataFrame) -> None:
    """Check that machine IDs are valid positive integers."""

    if not pd.api.types.is_integer_dtype(
        data["machine_id"]
    ):
        raise ValueError(
            "machine_id must contain integer values."
        )

    if (data["machine_id"] <= 0).any():
        raise ValueError(
            "machine_id must contain positive values."
        )

def check_sensor_values(data: pd.DataFrame) -> None:
    """Check that sensor values are finite."""

    for column in SENSOR_COLUMNS:
        values = data[column].to_numpy()

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values detected in "
                f"{column}."
            )
        
def detect_outliers_iqr(
    data: pd.DataFrame,
) -> dict[str, int]:
    """
    Detect potential outliers in continuous sensor features
    using the IQR method.

    Returns the number of potential outliers for each
    continuous numerical feature.
    """

    outlier_counts = {}

    for column in SENSOR_COLUMNS:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = data[
            (data[column] < lower_bound)
            | (data[column] > upper_bound)
        ]

        outlier_counts[column] = len(outliers)

    return outlier_counts


def calculate_iqr_bounds(
    data: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """
    Calculate IQR-based lower and upper bounds
    for continuous sensor features.
    """

    bounds = {}

    for column in SENSOR_COLUMNS:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        bounds[column] = {
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }

    return bounds

def flag_iqr_outliers(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag potential IQR outliers in continuous sensor features.

    Outliers are flagged for investigation rather than
    automatically removed or modified.
    """

    data = data.copy()

    bounds = calculate_iqr_bounds(data)

    data["has_potential_outlier"] = False

    for column, column_bounds in bounds.items():
        lower_bound = column_bounds["lower_bound"]
        upper_bound = column_bounds["upper_bound"]

        outlier_mask = (
            (data[column] < lower_bound)
            | (data[column] > upper_bound)
        )

        data.loc[
            outlier_mask,
            "has_potential_outlier"
        ] = True

    return data

def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Run all basic data-quality checks."""

    check_required_columns(data)
    check_data_types(data)
    check_timestamp(data)
    check_machine_ids(data)  
    check_missing_values(data)
    check_duplicate_records(data) 
    check_numeric_values(data)
    check_sensor_values(data)   
    check_failure_values(data) 
 
    return data 
