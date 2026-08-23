import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from zenml import step


@step
def train_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    RandomForestClassifier,
    pd.DataFrame,
    pd.Series,
    dict[str, float],
]:
    """Train, evaluate, and track the baseline Random Forest model."""

    # ---------------------------------------------------------
    # 1. Configure MLflow
    # ---------------------------------------------------------
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Predictive Maintenance")

    # ---------------------------------------------------------
    # 2. Split data into training and testing sets
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ---------------------------------------------------------
    # 3. Model configuration
    # ---------------------------------------------------------
    n_estimators = 100
    random_state = 42
    class_weight = "balanced"

    # ---------------------------------------------------------
    # 4. Start MLflow run
    # ---------------------------------------------------------
    with mlflow.start_run(
        run_name="predictive-maintenance-random-forest"
    ):

        # Log model parameters.
        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "random_state": random_state,
            "class_weight": class_weight,
            "test_size": 0.2,
        })

        # -----------------------------------------------------
        # 5. Create the model
        # -----------------------------------------------------
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
        )

        # -----------------------------------------------------
        # 6. Train the model
        # -----------------------------------------------------
        model.fit(X_train, y_train)

        # -----------------------------------------------------
        # 7. Evaluate on unseen test data
        # -----------------------------------------------------
        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        # Store metrics as a dictionary.
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

        # -----------------------------------------------------
        # 8. Log metrics to MLflow
        # -----------------------------------------------------
        mlflow.log_metrics(metrics)

        # -----------------------------------------------------
        # 9. Log trained model to MLflow
        # -----------------------------------------------------
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
        )

        # -----------------------------------------------------
        # 10. Display training results
        # -----------------------------------------------------
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    # ---------------------------------------------------------
    # 11. Return ZenML artifacts
    # ---------------------------------------------------------
    return model, X_test, y_test, metrics