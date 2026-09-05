import pandas as pd
from sklearn.compose import ColumnTransformer
from zenml import step

from src.feature_pipeline import (
    fit_and_transform_training_features,
)


@step(
    name="preprocess_engineered_features",
    enable_artifact_metadata=True,
)
def preprocess_engineered_features(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, ColumnTransformer]:
    """
    Fit the feature preprocessing pipeline on the
    supplied training features and transform them.

    Numerical features are scaled and machine IDs are
    one-hot encoded.
    """

    X_transformed, preprocessor = (
        fit_and_transform_training_features(X)
    )

    print(
        f"Original feature shape: {X.shape}"
    )

    print(
        f"Transformed feature shape: "
        f"{X_transformed.shape}"
    )

    print(
        f"Transformed feature names:\n"
        f"{list(X_transformed.columns)}"
    )

    return X_transformed, preprocessor