import numpy as np
import pandas as pd
import pytest

from src.data_validation import (
    check_required_columns,
    check_missing_values,
    check_duplicate_records,
    check_numeric_values,
    check_failure_values,
    check_data_types,
    check_machine_ids,
    flag_iqr_outliers,
)


def create_valid_data():
    """Create a small valid test dataset."""

    return pd.DataFrame({
        "timestamp": pd.to_datetime(
            ["2024-01-01 00:00:00"]
        ),
        "machine_id": [1],
        "ambient_temperature": [25.0],
        "humidity": [50.0],
        "temperature": [70.0],
        "vibration": [0.2],
        "pressure": [30.0],
        "rotational_speed": [1500.0],
        "torque": [45.0],
        "operating_hours": [1000.0],
        "failure": [0],
    })


def test_required_columns():
    data = create_valid_data()

    check_required_columns(data)


def test_missing_values_are_detected():
    data = create_valid_data()

    data.loc[0, "temperature"] = np.nan

    with pytest.raises(ValueError):
        check_missing_values(data)


def test_duplicate_records_are_detected():
    data = create_valid_data()

    data = pd.concat(
        [data, data],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        check_duplicate_records(data)


def test_non_finite_values_are_detected():
    data = create_valid_data()

    data.loc[0, "temperature"] = np.inf

    with pytest.raises(ValueError):
        check_numeric_values(data)


def test_invalid_failure_value_is_detected():
    data = create_valid_data()

    data.loc[0, "failure"] = 2

    with pytest.raises(ValueError):
        check_failure_values(data)


def test_data_types_are_valid():
    data = create_valid_data()

    check_data_types(data)


def test_invalid_data_type_is_detected():
    data = create_valid_data()

    data["temperature"] = (
        data["temperature"].astype(str)
    )

    with pytest.raises(ValueError):
        check_data_types(data)


def test_machine_ids_are_valid():
    data = create_valid_data()

    check_machine_ids(data)


def test_invalid_machine_id_is_detected():
    data = create_valid_data()

    data.loc[0, "machine_id"] = -1

    with pytest.raises(ValueError):
        check_machine_ids(data)


def test_iqr_outliers_are_flagged():
    """Verify that an extreme sensor value is flagged."""

    data = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-01-01 02:00:00",
            "2024-01-01 03:00:00",
            "2024-01-01 04:00:00",
            "2024-01-01 05:00:00",
            "2024-01-01 06:00:00",
            "2024-01-01 07:00:00",
            "2024-01-01 08:00:00",
            "2024-01-01 09:00:00",
            "2024-01-01 10:00:00",
        ]),
        "machine_id": [1] * 11,
        "ambient_temperature": [25.0] * 11,
        "humidity": [50.0] * 11,
        "temperature": [
            70.0,
            71.0,
            72.0,
            73.0,
            74.0,
            75.0,
            76.0,
            77.0,
            78.0,
            79.0,
            1000.0,
        ],
        "vibration": [0.2] * 11,
        "pressure": [30.0] * 11,
        "rotational_speed": [1500.0] * 11,
        "torque": [45.0] * 11,
        "operating_hours": [1000.0] * 11,
        "failure": [0] * 11,
    })

    flagged_data = flag_iqr_outliers(data)

    assert "has_potential_outlier" in flagged_data.columns

    assert bool(
        flagged_data.loc[
            10,
            "has_potential_outlier"
        ]
    ) is True