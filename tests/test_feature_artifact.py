import pandas as pd
import pytest

from src.feature_pipeline import (
    prepare_features,
    fit_and_transform_training_features,
    transform_model_features,
)
from src.feature_artifact import (
    save_feature_preprocessor,
    load_feature_preprocessor,
    transform_with_saved_preprocessor,
)


def create_sample_data():
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
            "failure": [
                0,
                0,
                1,
                0,
                1,
                0,
            ],
        }
    )


def test_fitted_preprocessor_can_be_saved(tmp_path):
    data = create_sample_data()

    X, _ = prepare_features(data)

    _, preprocessor = (
        fit_and_transform_training_features(X)
    )

    path = (
        tmp_path
        / "preprocessor.joblib"
    )

    saved_path = save_feature_preprocessor(
        preprocessor,
        path,
    )

    assert saved_path.exists()


def test_saved_preprocessor_can_be_loaded(tmp_path):
    data = create_sample_data()

    X, _ = prepare_features(data)

    _, preprocessor = (
        fit_and_transform_training_features(X)
    )

    path = (
        tmp_path
        / "preprocessor.joblib"
    )

    save_feature_preprocessor(
        preprocessor,
        path,
    )

    loaded_preprocessor = (
        load_feature_preprocessor(path)
    )

    assert loaded_preprocessor is not None


def test_loaded_preprocessor_produces_same_features(
    tmp_path,
):
    data = create_sample_data()

    X, _ = prepare_features(data)

    original_transformed, preprocessor = (
        fit_and_transform_training_features(X)
    )

    path = (
        tmp_path
        / "preprocessor.joblib"
    )

    save_feature_preprocessor(
        preprocessor,
        path,
    )

    loaded_transformed = (
        transform_with_saved_preprocessor(
            X,
            path,
        )
    )

    pd.testing.assert_frame_equal(
        original_transformed,
        loaded_transformed,
    )


def test_saved_preprocessor_handles_unknown_machine(
    tmp_path,
):
    data = create_sample_data()

    X, _ = prepare_features(data)

    _, preprocessor = (
        fit_and_transform_training_features(X)
    )

    path = (
        tmp_path
        / "preprocessor.joblib"
    )

    save_feature_preprocessor(
        preprocessor,
        path,
    )

    production_data = X.copy()

    production_data.loc[
        production_data.index[0],
        "machine_id",
    ] = 999

    transformed = (
        transform_with_saved_preprocessor(
            production_data,
            path,
        )
    )

    assert transformed.shape[0] == len(
        production_data
    )

    assert transformed.shape[1] == (
        len(
            preprocessor.get_feature_names_out()
        )
    )


def test_unfitted_preprocessor_cannot_be_saved(
    tmp_path,
):
    from src.feature_preprocessing import (
        create_feature_preprocessor,
    )

    preprocessor = (
        create_feature_preprocessor()
    )

    path = (
        tmp_path
        / "preprocessor.joblib"
    )

    with pytest.raises(ValueError):
        save_feature_preprocessor(
            preprocessor,
            path,
        )