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

from configs.pipeline_config import PipelineConfig



@step(
    name="train_random_forest",
    enable_artifact_metadata=True,
)
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

    # Load centralized pipeline configuration.
    config = PipelineConfig()

    # Configure MLflow using centralized configuration.
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    # Split data using configured parameters.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    # Start an MLflow experiment run.
    with mlflow.start_run(
        run_name="predictive-maintenance-random-forest"
    ):

        # Log model configuration to MLflow.
        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": config.n_estimators,
            "random_state": config.random_state,
            "class_weight": config.class_weight,
            "test_size": config.test_size,
        })

        # Create the model from centralized configuration.
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            class_weight=config.class_weight,
        )

        # Train only on training data.
        model.fit(X_train, y_train)

        # Evaluate on unseen test data.
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

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

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

        # Log evaluation metrics to MLflow.
        mlflow.log_metrics(metrics)

        # Log trained model to MLflow.
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    return model, X_test, y_test, metrics
