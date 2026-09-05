import pandas as pd

from src.feature_preprocessing_step import (
    preprocess_engineered_features,
)


def test_preprocess_engineered_features_step():
    X = pd.DataFrame(
        {
            "machine_id": [1, 2, 1, 3],
            "ambient_temperature": [
                25.0,
                26.0,
                25.5,
                27.0,
            ],
            "humidity": [
                50.0,
                55.0,
                52.0,
                60.0,
            ],
            "temperature": [
                40.0,
                42.0,
                41.0,
                45.0,
            ],
            "vibration": [
                1.0,
                1.2,
                1.1,
                1.3,
            ],
            "pressure": [
                100.0,
                101.0,
                100.5,
                102.0,
            ],
            "rotational_speed": [
                1500,
                1550,
                1520,
                1600,
            ],
            "torque": [
                20.0,
                21.0,
                20.5,
                22.0,
            ],
            "operating_hours": [
                100,
                110,
                120,
                130,
            ],
            "hour": [0, 1, 2, 3],
            "day_of_week": [0, 0, 0, 0],
            "month": [1, 1, 1, 1],
            "is_weekend": [0, 0, 0, 0],
            "temperature_difference": [
                15.0,
                16.0,
                15.5,
                18.0,
            ],
            "mechanical_load": [
                30000,
                32550,
                31160,
                35200,
            ],
            "operating_hours_squared": [
                10000,
                12100,
                14400,
                16900,
            ],
        }
    )

    X_transformed, preprocessor = (
        preprocess_engineered_features(X)
    )

    assert X_transformed.shape == (4, 18)
    assert preprocessor is not None