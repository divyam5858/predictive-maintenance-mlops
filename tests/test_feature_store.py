import pandas as pd
import pytest

from src.feature_store import FeatureStore


def test_register_feature_set(tmp_path):
    registry_path = (
        tmp_path
        / "feature_registry.json"
    )

    store = FeatureStore(registry_path)

    store.register_feature_set(
        name="predictive_maintenance_features",
        version="v1",
        features=[
            "temperature",
            "vibration",
            "pressure",
        ],
        description="Initial sensor feature set.",
    )

    feature_set = store.get_feature_set(
        "predictive_maintenance_features",
        "v1",
    )

    assert feature_set["features"] == [
        "temperature",
        "vibration",
        "pressure",
    ]


def test_feature_set_versioning(tmp_path):
    registry_path = (
        tmp_path
        / "feature_registry.json"
    )

    store = FeatureStore(registry_path)

    store.register_feature_set(
        name="predictive_maintenance_features",
        version="v1",
        features=[
            "temperature",
            "vibration",
        ],
    )

    store.register_feature_set(
        name="predictive_maintenance_features",
        version="v2",
        features=[
            "temperature",
            "vibration",
            "temperature_difference",
        ],
    )

    v1 = store.get_feature_set(
        "predictive_maintenance_features",
        "v1",
    )

    v2 = store.get_feature_set(
        "predictive_maintenance_features",
        "v2",
    )

    assert len(v1["features"]) == 2
    assert len(v2["features"]) == 3

    assert (
        "temperature_difference"
        not in v1["features"]
    )

    assert (
        "temperature_difference"
        in v2["features"]
    )


def test_duplicate_version_is_rejected(tmp_path):
    registry_path = (
        tmp_path
        / "feature_registry.json"
    )

    store = FeatureStore(registry_path)

    store.register_feature_set(
        name="maintenance_features",
        version="v1",
        features=["temperature"],
    )

    with pytest.raises(ValueError):
        store.register_feature_set(
            name="maintenance_features",
            version="v1",
            features=["temperature"],
        )


def test_materialize_features(tmp_path):
    registry_path = (
        tmp_path
        / "feature_registry.json"
    )

    store = FeatureStore(registry_path)

    store.register_feature_set(
        name="maintenance_features",
        version="v1",
        features=[
            "temperature",
            "vibration",
            "pressure",
        ],
    )

    data = pd.DataFrame(
        {
            "temperature": [40.0, 42.0],
            "vibration": [1.1, 1.2],
            "pressure": [100.0, 101.0],
            "humidity": [50.0, 55.0],
        }
    )

    materialized = store.materialize_features(
        data,
        "maintenance_features",
        "v1",
    )

    assert list(materialized.columns) == [
        "temperature",
        "vibration",
        "pressure",
    ]

    assert materialized.shape == (2, 3)


def test_missing_feature_is_rejected(tmp_path):
    registry_path = (
        tmp_path
        / "feature_registry.json"
    )

    store = FeatureStore(registry_path)

    store.register_feature_set(
        name="maintenance_features",
        version="v1",
        features=[
            "temperature",
            "vibration",
            "pressure",
        ],
    )

    data = pd.DataFrame(
        {
            "temperature": [40.0],
            "vibration": [1.1],
        }
    )

    with pytest.raises(ValueError):
        store.materialize_features(
            data,
            "maintenance_features",
            "v1",
        )