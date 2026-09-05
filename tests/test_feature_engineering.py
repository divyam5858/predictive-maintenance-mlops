import pandas as pd
import pytest

from src.feature_engineering import (
    create_sensor_features,
)


def create_sample_data():
    """Create sample predictive-maintenance data."""

    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-01 10:00:00",
            "2024-01-06 15:00:00",
        ]),
        "machine_id": [1, 2],
        "ambient_temperature": [25.0, 30.0],
        "humidity": [50.0, 60.0],
        "temperature": [75.0, 80.0],
        "vibration": [0.2, 0.4],
        "pressure": [30.0, 35.0],
        "rotational_speed": [1500.0, 2000.0],
        "torque": [40.0, 50.0],
        "operating_hours": [1000.0, 2000.0],
        "failure": [0, 1],
    })


def test_sensor_features_are_created():
    """Verify that engineered features are created."""

    data = create_sample_data()

    result = create_sensor_features(data)

    expected_features = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "temperature_difference",
        "mechanical_load",
        "operating_hours_squared",
    ]

    for feature in expected_features:
        assert feature in result.columns


def test_temperature_difference():
    """Verify temperature difference calculation."""

    data = create_sample_data()

    result = create_sensor_features(data)

    assert result.loc[
        0,
        "temperature_difference"
    ] == 50.0


def test_mechanical_load():
    """Verify mechanical load calculation."""

    data = create_sample_data()

    result = create_sensor_features(data)

    assert result.loc[
        0,
        "mechanical_load"
    ] == 60000.0


def test_time_features():
    """Verify timestamp-derived features."""

    data = create_sample_data()

    result = create_sensor_features(data)

    # 2024-01-01 is Monday.
    assert result.loc[0, "hour"] == 10
    assert result.loc[0, "day_of_week"] == 0
    assert result.loc[0, "month"] == 1
    assert result.loc[0, "is_weekend"] == 0

    # 2024-01-06 is Saturday.
    assert result.loc[1, "is_weekend"] == 1


def test_operating_hours_squared():
    """Verify nonlinear operating-hours feature."""

    data = create_sample_data()

    result = create_sensor_features(data)

    assert result.loc[
        0,
        "operating_hours_squared"
    ] == 1_000_000.0


def test_original_data_is_not_modified():
    """Verify feature engineering does not mutate input."""

    data = create_sample_data()

    original_columns = list(data.columns)

    create_sensor_features(data)

    assert list(data.columns) == original_columns


def test_missing_required_column_is_rejected():
    """Verify missing input columns are detected."""

    data = create_sample_data()

    data = data.drop(
        columns=["temperature"]
    )

    with pytest.raises(ValueError):
        create_sensor_features(data)