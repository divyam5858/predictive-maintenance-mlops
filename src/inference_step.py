import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from zenml import step


@step(
    name="predict_machine_failure",
    enable_artifact_metadata=True,
)
def predict_failure(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.Series:
    """
    Generate predictions using transformed test features.
    """

    predictions = model.predict(X_test)

    print(
        f"Prediction samples: {len(predictions)}"
    )

    print(
        f"Actual test samples: {len(y_test)}"
    )

    print(
        f"Predicted failure distribution:\n"
        f"{pd.Series(predictions).value_counts()}"
    )

    return pd.Series(
        predictions,
        index=X_test.index,
        name="prediction",
    )