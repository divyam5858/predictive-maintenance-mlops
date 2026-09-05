import pandas as pd
import pytest

from src.feature_pipeline import (
    prepare_features,
    fit_and_transform_training_features,
    transform_model_features,
)


@pytest.fixture
def sample_sensor_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=6,
                freq="h",
            ),
            "machine_id": [1, 2, 1, 3, 2, 3],
            "ambient_temperature": [
                25.0,
                26.0,
                25.5,
                27.0,
                26.5,
                28.0,
            ],
            "humidity": [
                50.0,
                55.0,
                52.0,
                60.0,
                58.0,
                62.0,
            ],
            "temperature": [
                40.0,
                42.0,
                41.0,
                45.0,
                44.0,
                47.0,
            ],
            "vibration": [
                1.0,
                1.2,
                1.1,
                1.3,
                1.2,
                1.4,
            ],
            "pressure": [
                100.0,
                101.0,
                100.5,
                102.0,
                101.5,
                103.0,
            ],
            "rotational_speed": [
                1500,
                1550,
                1520,
                1600,
                1580,
                1650,
            ],
            "torque": [
                20.0,
                21.0,
                20.5,
                22.0,
                21.5,
                23.0,
            ],
            "operating_hours": [
                100,
                110,
                120,
                130,
                140,
                150,
            ],
            "failure": [0, 0, 1, 0, 1, 0],
        }
    )


def test_prepare_features(sample_sensor_data):
    X, y = prepare_features(sample_sensor_data)

    assert len(X) == 6
    assert len(y) == 6

    assert "failure" not in X.columns
    assert "timestamp" not in X.columns

    assert "temperature_difference" in X.columns
    assert "mechanical_load" in X.columns
    assert "operating_hours_squared" in X.columns


def test_prepare_features_preserves_target(sample_sensor_data):
    _, y = prepare_features(sample_sensor_data)

    assert y.tolist() == [0, 0, 1, 0, 1, 0]


def test_training_features_are_transformed(sample_sensor_data):
    X, _ = prepare_features(sample_sensor_data)

    X_transformed, preprocessor = (
        fit_and_transform_training_features(X)
    )

    assert X_transformed.shape[0] == 6
    assert X_transformed.shape[1] == 18

    assert preprocessor is not None


def test_test_features_reuse_fitted_preprocessor(
    sample_sensor_data,
):
    X, _ = prepare_features(sample_sensor_data)

    X_train = X.iloc[:4]
    X_test = X.iloc[4:]

    X_train_transformed, preprocessor = (
        fit_and_transform_training_features(X_train)
    )

    X_test_transformed = transform_model_features(
        preprocessor,
        X_test,
    )

    assert X_train_transformed.shape[1] == 18
    assert X_test_transformed.shape[1] == 18


def test_unknown_machine_is_supported(sample_sensor_data):
    X, _ = prepare_features(sample_sensor_data)

    X_train = X.iloc[:4].copy()
    X_test = X.iloc[4:].copy()

    X_test.loc[X_test.index[0], "machine_id"] = 99

    _, preprocessor = (
        fit_and_transform_training_features(X_train)
    )

    X_test_transformed = transform_model_features(
        preprocessor,
        X_test,
    )

    assert X_test_transformed.shape[1] == 18