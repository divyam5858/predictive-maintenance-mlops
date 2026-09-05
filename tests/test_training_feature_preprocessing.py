import pandas as pd

from src.training_step import train_model


def create_sample_features():
    X = pd.DataFrame(
        {
            "machine_id": [1, 2, 1, 3, 2, 3, 1, 2],
            "ambient_temperature": [
                25.0,
                26.0,
                25.5,
                27.0,
                26.5,
                28.0,
                25.2,
                26.8,
            ],
            "humidity": [
                50.0,
                55.0,
                52.0,
                60.0,
                58.0,
                62.0,
                51.0,
                57.0,
            ],
            "temperature": [
                40.0,
                42.0,
                41.0,
                45.0,
                44.0,
                47.0,
                40.5,
                43.0,
            ],
            "vibration": [
                1.0,
                1.2,
                1.1,
                1.3,
                1.2,
                1.4,
                1.05,
                1.25,
            ],
            "pressure": [
                100.0,
                101.0,
                100.5,
                102.0,
                101.5,
                103.0,
                100.2,
                101.8,
            ],
            "rotational_speed": [
                1500,
                1550,
                1520,
                1600,
                1580,
                1650,
                1510,
                1570,
            ],
            "torque": [
                20.0,
                21.0,
                20.5,
                22.0,
                21.5,
                23.0,
                20.2,
                21.2,
            ],
            "operating_hours": [
                100,
                110,
                120,
                130,
                140,
                150,
                160,
                170,
            ],
            "hour": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ],
            "day_of_week": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "month": [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            "is_weekend": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "temperature_difference": [
                15.0,
                16.0,
                15.5,
                18.0,
                17.5,
                19.0,
                15.3,
                16.2,
            ],
            "mechanical_load": [
                30000,
                32550,
                31160,
                35200,
                33970,
                37950,
                30502,
                33384,
            ],
            "operating_hours_squared": [
                10000,
                12100,
                14400,
                16900,
                19600,
                22500,
                25600,
                28900,
            ],
        }
    )

    y = pd.Series(
        [0, 0, 1, 0, 1, 0, 0, 1],
        name="failure",
    )

    return X, y


def test_training_step_uses_transformed_features():
    X, y = create_sample_features()

    (
        model,
        preprocessor,
        X_test,
        y_test,
        metrics,
    ) = train_model(X, y)

    assert model is not None
    assert preprocessor is not None

    assert X_test.shape[1] >= 17
    assert X_test.shape[1] <= 18

    assert len(X_test) == len(y_test)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics