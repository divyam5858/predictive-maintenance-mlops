import pandas as pd
import pytest

from src.feature_engineering import create_sensor_features
from src.feature_selection import select_features
from src.feature_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    create_feature_preprocessor,
    fit_feature_preprocessor,
    transform_features,
)


def create_sample_data():
    """Create sample predictive-maintenance data."""

    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-01 10:00:00",
            "2024-01-06 15:00:00",
            "2024-02-10 12:00:00",
            "2024-03-11 08:00:00",
        ]),
        "machine_id": [1, 2, 1, 3],
        "ambient_temperature": [25.0, 30.0, 27.0, 24.0],
        "humidity": [50.0, 60.0, 55.0, 45.0],
        "temperature": [75.0, 80.0, 78.0, 70.0],
        "vibration": [0.2, 0.4, 0.3, 0.1],
        "pressure": [30.0, 35.0, 32.0, 29.0],
        "rotational_speed": [
            1500.0,
            2000.0,
            1800.0,
            1400.0,
        ],
        "torque": [40.0, 50.0, 45.0, 35.0],
        "operating_hours": [
            1000.0,
            2000.0,
            1500.0,
            800.0,
        ],
        "failure": [0, 1, 0, 0],
    })


def prepare_features():
    """Create engineered and selected features."""

    data = create_sample_data()

    engineered_data = create_sensor_features(data)

    X, y = select_features(engineered_data)

    return X, y


def test_preprocessor_contains_expected_feature_groups():
    """Verify numerical and categorical feature groups."""

    preprocessor = create_feature_preprocessor()

    transformer_names = [
        name
        for name, _, _ in preprocessor.transformers
    ]

    assert "numerical" in transformer_names
    assert "categorical" in transformer_names

    assert len(NUMERICAL_FEATURES) == 15
    assert len(CATEGORICAL_FEATURES) == 1


def test_preprocessor_can_be_fitted():
    """Verify that the preprocessor can fit training data."""

    X, _ = prepare_features()

    preprocessor = fit_feature_preprocessor(X)

    assert preprocessor is not None
    assert hasattr(
        preprocessor,
        "transformers_",
    )


def test_features_are_transformed():
    """Verify that selected features are transformed."""

    X, _ = prepare_features()

    preprocessor = fit_feature_preprocessor(X)

    transformed = transform_features(
        preprocessor,
        X,
    )

    assert isinstance(
        transformed,
        pd.DataFrame,
    )

    assert transformed.shape[0] == X.shape[0]


def test_machine_id_is_one_hot_encoded():
    """Verify machine IDs become categorical features."""

    X, _ = prepare_features()

    preprocessor = fit_feature_preprocessor(X)

    transformed = transform_features(
        preprocessor,
        X,
    )

    machine_columns = [
        column
        for column in transformed.columns
        if column.startswith("machine_id_")
    ]

    assert len(machine_columns) == 3


def test_numerical_features_are_scaled():
    """Verify numerical features are standardized."""

    X, _ = prepare_features()

    preprocessor = fit_feature_preprocessor(X)

    transformed = transform_features(
        preprocessor,
        X,
    )

    numerical_data = transformed[
        NUMERICAL_FEATURES
    ]

    means = numerical_data.mean()

    for mean in means:
        assert abs(mean) < 1e-10


def test_unknown_machine_id_is_handled():
    """Verify unseen machine IDs do not break transformation."""

    X_train, _ = prepare_features()

    preprocessor = fit_feature_preprocessor(
        X_train
    )

    X_new = X_train.iloc[[0]].copy()

    X_new["machine_id"] = 999

    transformed = transform_features(
        preprocessor,
        X_new,
    )

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == 18


def test_missing_feature_is_rejected():
    """Verify required features are checked by the preprocessor."""

    X, _ = prepare_features()

    X = X.drop(
        columns=["temperature"]
    )

    preprocessor = create_feature_preprocessor()

    with pytest.raises(
        ValueError,
    ):
        preprocessor.fit(X)