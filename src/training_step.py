import os

import mlflow
import pandas as pd

from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from zenml import step

from configs.pipeline_config import PipelineConfig
from src.feature_pipeline import (
    fit_and_transform_training_features,
    transform_model_features,
)


@step(
    name="train_random_forest",
    enable_artifact_metadata=True,
    enable_cache=False,
)
def train_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    RandomForestClassifier,
    object,
    pd.DataFrame,
    pd.Series,
    dict,
]:
    """
    Split the data, fit feature preprocessing on the
    training partition only, train the Random Forest model,
    evaluate it on transformed test data, and log
    experiment artifacts to MLflow.
    """

    config = PipelineConfig()

    # Configure MLflow.
    mlflow.set_tracking_uri(
        config.mlflow_tracking_uri
    )

    mlflow.set_experiment(
        config.mlflow_experiment_name
    )

    # Split the feature-selected data into training
    # and test partitions.
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y,
        )
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Test samples: {len(X_test)}"
    )

    # Fit the feature preprocessing pipeline using
    # training data only.
    X_train_transformed, preprocessor = (
        fit_and_transform_training_features(
            X_train
        )
    )

    # Reuse the fitted preprocessor to transform
    # the test data.
    X_test_transformed = transform_model_features(
        preprocessor,
        X_test,
    )

    print(
        f"Transformed training shape: "
        f"{X_train_transformed.shape}"
    )

    print(
        f"Transformed test shape: "
        f"{X_test_transformed.shape}"
    )

    # Start an MLflow run for experiment tracking.
    with mlflow.start_run(
        run_name="predictive-maintenance-random-forest"
    ):

        # Log model configuration.
        mlflow.log_param(
            "model_type",
            "RandomForestClassifier",
        )

        mlflow.log_param(
            "test_size",
            config.test_size,
        )

        mlflow.log_param(
            "random_state",
            config.random_state,
        )

        mlflow.log_param(
            "n_estimators",
            config.n_estimators,
        )

        mlflow.log_param(
            "class_weight",
            config.class_weight,
        )

        # Record the number of features after
        # scaling and categorical encoding.
        mlflow.log_param(
            "feature_count",
            X_train_transformed.shape[1],
        )

        # Create the Random Forest model.
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            class_weight=config.class_weight,
        )

        # Train the model using transformed training data.
        model.fit(
            X_train_transformed,
            y_train,
        )

        # Generate predictions on transformed test data.
        predictions = model.predict(
            X_test_transformed
        )

        # Infer the MLflow model signature from
        # the transformed test features and predictions.
        signature = infer_signature(
            X_test_transformed,
            predictions,
        )

        # Calculate evaluation metrics.
        metrics = {
            "accuracy": accuracy_score(
                y_test,
                predictions,
            ),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
        }

        # Log evaluation metrics to MLflow.
        for metric_name, value in metrics.items():
            mlflow.log_metric(
                metric_name,
                value,
            )

        # Create a temporary directory for experiment artifacts.
        artifact_dir = "mlflow_artifacts"

        os.makedirs(
            artifact_dir,
            exist_ok=True,
        )

        # Create a detailed classification report.
        classification_report_path = os.path.join(
            artifact_dir,
            "classification_report.txt",
        )

        with open(
            classification_report_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                classification_report(
                    y_test,
                    predictions,
                    digits=4,
                )
            )

        # Create a confusion matrix artifact.
        confusion_matrix_path = os.path.join(
            artifact_dir,
            "confusion_matrix.csv",
        )

        confusion_matrix_values = confusion_matrix(
            y_test,
            predictions,
        )

        confusion_matrix_df = pd.DataFrame(
            confusion_matrix_values,
            index=[
                "actual_0",
                "actual_1",
            ],
            columns=[
                "predicted_0",
                "predicted_1",
            ],
        )

        confusion_matrix_df.to_csv(
            confusion_matrix_path
        )

        # Record the transformed feature names.
        feature_names_path = os.path.join(
            artifact_dir,
            "feature_names.txt",
        )

        feature_names = X_train_transformed.columns

        with open(
            feature_names_path,
            "w",
            encoding="utf-8",
        ) as file:
            for feature_name in feature_names:
                file.write(
                    f"{feature_name}\n"
                )

        # Log the generated artifacts to MLflow.
        mlflow.log_artifacts(
            artifact_dir
        )

        # Log the trained model with its input/output signature.
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
        )

        # Display metrics in the pipeline output.
        print(
            f"Accuracy: {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall: {metrics['recall']:.4f}"
        )

        print(
            f"F1 Score: {metrics['f1']:.4f}"
        )

    # Return the model, fitted preprocessor, transformed
    # test data, test target, and evaluation metrics.
    return (
        model,
        preprocessor,
        X_test_transformed,
        y_test,
        metrics,
    )