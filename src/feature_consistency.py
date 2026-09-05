import pandas as pd
from sklearn.compose import ColumnTransformer

from src.feature_pipeline import (
    prepare_features,
    fit_and_transform_training_features,
    transform_model_features,
)


def build_training_features(
    raw_training_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    ColumnTransformer,
]:
    """
    Build model-ready training features.

    Feature engineering and feature selection are applied
    before fitting the preprocessing pipeline.

    The preprocessor is fitted using training data only.
    """

    X_train, y_train = prepare_features(
        raw_training_data
    )

    X_train_transformed, preprocessor = (
        fit_and_transform_training_features(
            X_train
        )
    )

    return (
        X_train_transformed,
        y_train,
        preprocessor,
    )


def build_production_features(
    raw_production_data: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> pd.DataFrame:
    """
    Build model-ready production features using the
    already-fitted training preprocessor.
    """

    X_production, _ = prepare_features(
        raw_production_data
    )

    return transform_model_features(
        preprocessor,
        X_production,
    )


def validate_feature_consistency(
    training_features: pd.DataFrame,
    production_features: pd.DataFrame,
) -> bool:
    """
    Verify that training and production features have
    the same schema.
    """

    if list(training_features.columns) != list(
        production_features.columns
    ):
        return False

    if training_features.shape[1] != (
        production_features.shape[1]
    ):
        return False

    return True