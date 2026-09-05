from src.feature_store import (
    FeatureStore,
    register_predictive_maintenance_features,
)


def main():
    store = FeatureStore()

    feature_set_name = (
        "predictive_maintenance_features"
    )

    feature_set_version = "v1"

    existing_feature_sets = (
        store.list_feature_sets()
    )

    if (
        feature_set_name in existing_feature_sets
        and feature_set_version
        in existing_feature_sets[feature_set_name]
    ):
        print(
            f"{feature_set_name} "
            f"{feature_set_version} is already registered."
        )
    else:
        register_predictive_maintenance_features(
            store,
            feature_set_version,
        )

        print(
            f"Registered {feature_set_name} "
            f"{feature_set_version}."
        )

    feature_set = store.get_feature_set(
        feature_set_name,
        feature_set_version,
    )

    print(
        f"\nFeature set: {feature_set_name}"
    )

    print(
        f"Version: {feature_set_version}"
    )

    print(
        f"Feature count: "
        f"{len(feature_set['features'])}"
    )

    print(
        "Features:"
    )

    for feature in feature_set["features"]:
        print(f"- {feature}")


if __name__ == "__main__":
    main()