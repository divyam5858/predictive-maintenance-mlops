import pandas as pd
import pytest

from src.gx_validation import (
    validate_with_great_expectations,
)


def create_valid_data():
    """Create a valid sensor dataset for GX testing."""

    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-01-01 02:00:00",
        ]),
        "machine_id": [1, 2, 3],
        "ambient_temperature": [25.0, 26.0, 24.0],
        "humidity": [50.0, 55.0, 48.0],
        "temperature": [70.0, 72.0, 68.0],
        "vibration": [0.2, 0.3, 0.1],
        "pressure": [30.0, 31.0, 29.0],
        "rotational_speed": [1500.0, 1550.0, 1480.0],
        "torque": [45.0, 46.0, 44.0],
        "operating_hours": [1000.0, 1100.0, 900.0],
        "failure": [0, 1, 0],
    })


def test_gx_validation_passes_for_valid_data():
    """Valid data should pass all GX expectations."""

    data = create_valid_data()

    result = validate_with_great_expectations(data)

    assert result is True


def test_gx_rejects_invalid_failure_value():
    """GX should reject invalid target values."""

    data = create_valid_data()

    data.loc[0, "failure"] = 2

    with pytest.raises(ValueError):
        validate_with_great_expectations(data)


def test_gx_rejects_invalid_humidity():
    """GX should reject humidity outside 0-100."""

    data = create_valid_data()

    data.loc[0, "humidity"] = 150.0

    with pytest.raises(ValueError):
        validate_with_great_expectations(data)


def test_gx_rejects_negative_sensor_value():
    """GX should reject negative operational measurements."""

    data = create_valid_data()

    data.loc[0, "vibration"] = -1.0

    with pytest.raises(ValueError):
        validate_with_great_expectations(data)


def test_gx_rejects_missing_values():
    """GX should reject missing required values."""

    data = create_valid_data()

    data.loc[0, "temperature"] = None

    with pytest.raises(ValueError):
        validate_with_great_expectations(data) 