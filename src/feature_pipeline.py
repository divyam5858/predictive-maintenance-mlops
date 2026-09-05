import pandas as pd

from sklearn.compose import ColumnTransformer

from src.feature_engineering import create_sensor_features
from src.feature_selection import select_features
from src.feature_preprocessing import (
    fit_feature_preprocessor,
    transform_features,
)


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create and select features from raw sensor data.

    This function performs:
    1. Domain-driven feature engineering.
    2. Feature selection.
    3. Separation of model inputs and target.

    The returned X contains selected, untransformed features.
    """

    engineered_data = create_sensor_features(df)

    X, y = select_features(engineered_data)

    return X, y


def fit_and_transform_training_features(
    X_train: pd.DataFrame,
) -> tuple[pd.DataFrame, ColumnTransformer]:
    """
    Fit the feature preprocessor on training data and
    transform the training features.
    """

    preprocessor = fit_feature_preprocessor(X_train)

    X_train_transformed = transform_features(
        preprocessor,
        X_train,
    )

    return X_train_transformed, preprocessor


def transform_model_features(
    preprocessor: ColumnTransformer,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform validation, test, or production data using
    an already-fitted feature preprocessor.
    """

    return transform_features(
        preprocessor,
        data,
    )