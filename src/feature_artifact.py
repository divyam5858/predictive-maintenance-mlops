from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer


DEFAULT_PREPROCESSOR_PATH = Path(
    "artifacts/feature_store/"
    "predictive_maintenance_preprocessor.joblib"
)


def save_feature_preprocessor(
    preprocessor: ColumnTransformer,
    path: Path = DEFAULT_PREPROCESSOR_PATH,
) -> Path:
    """
    Save a fitted feature preprocessor to disk.

    The saved preprocessor contains the learned scaling
    parameters and categorical encoding configuration.
    """

    if not hasattr(
        preprocessor,
        "transformers_",
    ):
        raise ValueError(
            "Preprocessor must be fitted before saving."
        )

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        preprocessor,
        path,
    )

    return path


def load_feature_preprocessor(
    path: Path = DEFAULT_PREPROCESSOR_PATH,
) -> ColumnTransformer:
    """
    Load a previously fitted feature preprocessor.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor artifact not found: {path}"
        )

    preprocessor = joblib.load(path)

    return preprocessor


def transform_with_saved_preprocessor(
    data: pd.DataFrame,
    path: Path = DEFAULT_PREPROCESSOR_PATH,
) -> pd.DataFrame:
    """
    Load a saved preprocessor and transform new data.
    """

    preprocessor = load_feature_preprocessor(
        path
    )

    from src.feature_pipeline import (
        transform_model_features,
    )

    return transform_model_features(
        preprocessor,
        data,
    )