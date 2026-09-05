import pandas as pd
from zenml import step

from src.feature_pipeline import prepare_features


@step(
    name="engineer_and_select_features",
    enable_artifact_metadata=True,
)
def engineer_and_select_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create domain-driven features and select the
    model-ready feature set.

    This step performs feature engineering and feature
    selection while keeping preprocessing separate.
    """

    X, y = prepare_features(df)

    print(
        f"Selected feature shape: {X.shape}"
    )

    print(
        f"Selected features:\n"
        f"{list(X.columns)}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    print(
        f"Failure distribution:\n"
        f"{y.value_counts()}"
    )

    return X, y