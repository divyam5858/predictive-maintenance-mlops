import pandas as pd
import pytest

from src.feature_engineering import create_sensor_features
from src.feature_selection import (
    SELECTED_FEATURES,
    select_features,
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


def test_selected_features_are_returned():
    """Verify that the expected model features are selected."""

    data = create_sample_data()

    engineered_data = create_sensor_features(data)

    X, y = select_features(engineered_data)

    assert list(X.columns) == SELECTED_FEATURES


def test_target_is_separated():
    """Verify that failure is separated from model features."""

    data = create_sample_data()

    engineered_data = create_sensor_features(data)

    X, y = select_features(engineered_data)

    assert "failure" not in X.columns
    assert y.name == "failure"


def test_timestamp_is_removed_from_features():
    """Verify that raw timestamp is not passed to the model."""

    data = create_sample_data()

    engineered_data = create_sensor_features(data)

    X, _ = select_features(engineered_data)

    assert "timestamp" not in X.columns


def test_expected_feature_count():
    """Verify the selected feature count."""

    data = create_sample_data()

    engineered_data = create_sensor_features(data)

    X, y = select_features(engineered_data)

    assert X.shape[1] == 16
    assert y.shape[0] == data.shape[0]


def test_machine_id_is_retained():
    """Verify machine ID remains available for categorical encoding."""

    data = create_sample_data()

    engineered_data = create_sensor_features(data)

    X, _ = select_features(engineered_data)

    assert "machine_id" in X.columns


def test_missing_feature_is_rejected():
    """Verify missing engineered features are detected."""

    data = create_sample_data()

    engineered_data = create_sensor_features(data)

    engineered_data = engineered_data.drop(
        columns=["mechanical_load"]
    )

    with pytest.raises(ValueError):
        select_features(engineered_data)