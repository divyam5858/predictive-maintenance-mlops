from src.feature_store import (
    FeatureStore,
    register_predictive_maintenance_features,
)


def test_predictive_maintenance_feature_set(
    tmp_path,
):
    registry_path = (
        tmp_path
        / "feature_registry.json"
    )

    store = FeatureStore(registry_path)

    register_predictive_maintenance_features(
        store,
        "v1",
    )

    feature_set = store.get_feature_set(
        "predictive_maintenance_features",
        "v1",
    )

    expected_features = [
        "machine_id",
        "ambient_temperature",
        "humidity",
        "temperature",
        "vibration",
        "pressure",
        "rotational_speed",
        "torque",
        "operating_hours",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "temperature_difference",
        "mechanical_load",
        "operating_hours_squared",
    ]

    assert feature_set["features"] == (
        expected_features
    )

    assert len(
        feature_set["features"]
    ) == 16


def test_feature_set_persists_across_store_instances(
    tmp_path,
):
    registry_path = (
        tmp_path
        / "feature_registry.json"
    )

    store = FeatureStore(registry_path)

    register_predictive_maintenance_features(
        store,
        "v1",
    )

    new_store = FeatureStore(
        registry_path
    )

    feature_set = new_store.get_feature_set(
        "predictive_maintenance_features",
        "v1",
    )

    assert len(
        feature_set["features"]
    ) == 16


def test_feature_set_materializes_real_features(
    tmp_path,
):
    import pandas as pd

    registry_path = (
        tmp_path
        / "feature_registry.json"
    )

    store = FeatureStore(registry_path)

    register_predictive_maintenance_features(
        store,
        "v1",
    )

    data = pd.DataFrame(
        {
            feature: [1.0, 2.0]
            for feature in [
                "machine_id",
                "ambient_temperature",
                "humidity",
                "temperature",
                "vibration",
                "pressure",
                "rotational_speed",
                "torque",
                "operating_hours",
                "hour",
                "day_of_week",
                "month",
                "is_weekend",
                "temperature_difference",
                "mechanical_load",
                "operating_hours_squared",
            ]
        }
    )

    materialized = (
        store.materialize_features(
            data,
            "predictive_maintenance_features",
            "v1",
        )
    )

    assert materialized.shape == (2, 16)