import pandas as pd

from src.feature_consistency import (
    build_training_features,
    build_production_features,
    validate_feature_consistency,
)


def create_sample_data() -> pd.DataFrame:
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


def test_training_features_are_built():
    data = create_sample_data()

    X_train, y_train, preprocessor = (
        build_training_features(data)
    )

    assert len(X_train) == len(y_train)
    assert X_train.shape[1] == 18
    assert preprocessor is not None


def test_production_reuses_training_preprocessor():
    data = create_sample_data()

    _, _, preprocessor = build_training_features(
        data
    )

    production_features = build_production_features(
        data,
        preprocessor,
    )

    assert production_features.shape[1] == 18


def test_training_and_production_schema_matches():
    data = create_sample_data()

    training_features, _, preprocessor = (
        build_training_features(data)
    )

    production_features = build_production_features(
        data,
        preprocessor,
    )

    assert validate_feature_consistency(
        training_features,
        production_features,
    )


def test_unknown_machine_does_not_break_production():
    data = create_sample_data()

    training_features, _, preprocessor = (
        build_training_features(data)
    )

    production_data = data.copy()

    production_data.loc[
        production_data.index[0],
        "machine_id",
    ] = 99

    production_features = build_production_features(
        production_data,
        preprocessor,
    )

    assert validate_feature_consistency(
        training_features,
        production_features,
    )


def test_inconsistent_schema_is_detected():
    training_features = pd.DataFrame(
        {
            "feature_a": [1.0],
            "feature_b": [2.0],
        }
    )

    production_features = pd.DataFrame(
        {
            "feature_a": [1.0],
            "feature_c": [2.0],
        }
    )

    assert not validate_feature_consistency(
        training_features,
        production_features,
    )