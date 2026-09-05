import json
from pathlib import Path

import pandas as pd


DEFAULT_REGISTRY_PATH = Path(
    "artifacts/feature_store/feature_registry.json"
)


class FeatureStore:
    """
    Lightweight local feature store for the predictive
    maintenance project.

    The registry stores feature-set definitions and versions.
    """

    def __init__(
        self,
        registry_path: Path = DEFAULT_REGISTRY_PATH,
    ):
        self.registry_path = Path(registry_path)

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.registry_path.exists():
            with open(
                self.registry_path,
                "r",
                encoding="utf-8",
            ) as file:
                self.registry = json.load(file)
        else:
            self.registry = {
                "feature_sets": {}
            }

    def save(self) -> None:
        """Persist the feature registry to disk."""

        with open(
            self.registry_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.registry,
                file,
                indent=4,
            )

    def register_feature_set(
        self,
        name: str,
        version: str,
        features: list[str],
        description: str = "",
    ) -> None:
        """
        Register a new version of a feature set.
        """

        if not name:
            raise ValueError(
                "Feature set name cannot be empty."
            )

        if not version:
            raise ValueError(
                "Feature set version cannot be empty."
            )

        if not features:
            raise ValueError(
                "Feature set must contain at least one feature."
            )

        feature_sets = self.registry.setdefault(
            "feature_sets",
            {},
        )

        versions = feature_sets.setdefault(
            name,
            {},
        )

        if version in versions:
            raise ValueError(
                f"Feature set '{name}' version "
                f"'{version}' already exists."
            )

        versions[version] = {
            "features": features,
            "description": description,
        }

        self.save()

    def get_feature_set(
        self,
        name: str,
        version: str,
    ) -> dict:
        """
        Retrieve a specific feature-set version.
        """

        try:
            return self.registry[
                "feature_sets"
            ][name][version]
        except KeyError as exc:
            raise ValueError(
                f"Feature set '{name}' version "
                f"'{version}' was not found."
            ) from exc

    def list_feature_sets(self) -> dict:
        """
        Return all registered feature sets and versions.
        """

        return self.registry.get(
            "feature_sets",
            {},
        )

    def materialize_features(
        self,
        data: pd.DataFrame,
        name: str,
        version: str,
    ) -> pd.DataFrame:
        """
        Return only the features registered for a
        specific feature-set version.
        """

        feature_set = self.get_feature_set(
            name,
            version,
        )

        features = feature_set["features"]

        missing_features = [
            feature
            for feature in features
            if feature not in data.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing features: {missing_features}"
            )

        return data[features].copy()


def register_predictive_maintenance_features(
    store: FeatureStore,
    version: str = "v1",
) -> None:
    """
    Register the production feature set used by the
    predictive maintenance model.
    """

    features = [
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

    store.register_feature_set(
        name="predictive_maintenance_features",
        version=version,
        features=features,
        description=(
            "Versioned feature set for the "
            "predictive maintenance model."
        ),
    )