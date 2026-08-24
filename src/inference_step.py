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
) -> pd.DataFrame:
    """Generate predictions on unseen test data."""

    predictions = model.predict(X_test)

    results = pd.DataFrame({
        "actual_failure": y_test.values,
        "predicted_failure": predictions,
    })

    print("Inference completed on unseen test data.")
    print(f"Prediction samples: {len(results)}")
    print("\nPrediction distribution:")
    print(results["predicted_failure"].value_counts())

    return results