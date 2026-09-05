import pandas as pd

from src.feature_engineering_step import (
    engineer_and_select_features,
)


def test_engineer_and_select_features_step():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=4,
                freq="h",
            ),
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
            "failure": [0, 0, 1, 0],
        }
    )

    X, y = engineer_and_select_features(data)

    assert X.shape == (4, 16)
    assert y.shape == (4,)

    assert "timestamp" not in X.columns
    assert "failure" not in X.columns

    assert (
        "temperature_difference"
        in X.columns
    )

    assert (
        "mechanical_load"
        in X.columns
    )