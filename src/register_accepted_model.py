import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from configs.pipeline_config import PipelineConfig
from src.data_cleaning import clean_sensor_data
from src.data_ingestion import load_sensor_data
from src.data_validation import validate_data
from src.feature_pipeline import prepare_features
from src.feature_preprocessing import (
    create_feature_preprocessor,
)
from src.gx_validation import (
    validate_with_great_expectations,
)
from src.model_evaluation import evaluate_model
from src.model_training import create_random_forest


ACCEPTED_EVALUATION_RUN_ID = (
    "6db4b5b0f150490e8f2cbefd65ce47b9"
)

ACCEPTED_MODEL_PARAMETERS = {
    "n_estimators": 200,
    "min_samples_leaf": 2,
    "max_depth": None,
}

EXPECTED_METRICS = {
    "accuracy": 0.8950,
    "precision": 0.4090909090909091,
    "recall": 0.703125,
    "f1": 0.5172413793103449,
}


def build_accepted_model() -> Pipeline:
    """Create the accepted Random Forest pipeline."""

    config = PipelineConfig()

    model = create_random_forest(
        n_estimators=(
            ACCEPTED_MODEL_PARAMETERS[
                "n_estimators"
            ]
        ),
        random_state=config.random_state,
        class_weight=config.class_weight,
    )

    model.set_params(
        min_samples_leaf=(
            ACCEPTED_MODEL_PARAMETERS[
                "min_samples_leaf"
            ]
        ),
        max_depth=(
            ACCEPTED_MODEL_PARAMETERS[
                "max_depth"
            ]
        ),
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_feature_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )


def metrics_match_expected(
    metrics: dict[str, float],
    tolerance: float = 1e-10,
) -> bool:
    """Check that recreated metrics match Chapter 8 results."""

    for metric_name, expected_value in (
        EXPECTED_METRICS.items()
    ):
        actual_value = metrics[metric_name]

        if abs(
            actual_value - expected_value
        ) > tolerance:
            print(
                f"Metric mismatch for {metric_name}: "
                f"expected={expected_value:.10f}, "
                f"actual={actual_value:.10f}"
            )
            return False

    return True


def find_existing_version(
    client: MlflowClient,
    model_name: str,
):
    """Find an already registered version from this evaluation run."""

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    for version in versions:
        if (
            version.tags.get(
                "source_evaluation_run_id"
            )
            == ACCEPTED_EVALUATION_RUN_ID
        ):
            return version

    return None


def main() -> None:
    """Recreate, validate, log, and register the accepted model."""

    config = PipelineConfig()

    mlflow.set_tracking_uri(
        config.mlflow_tracking_uri
    )

    mlflow.set_experiment(
        config.mlflow_experiment_name
    )

    client = MlflowClient(
        tracking_uri=config.mlflow_tracking_uri
    )

    print("Loading raw sensor data...")

    data = load_sensor_data()

    cleaned_data = clean_sensor_data(
        data
    )

    print(
        "Running Pandas/NumPy validation..."
    )

    validate_data(
        cleaned_data
    )

    print(
        "Pandas/NumPy validation passed."
    )

    print(
        "Running Great Expectations validation..."
    )

    validate_with_great_expectations(
        cleaned_data
    )

    print(
        "Great Expectations validation passed."
    )

    print(
        "Preparing model features..."
    )

    X, y = prepare_features(
        cleaned_data
    )

    print(
        f"Selected feature shape: {X.shape}"
    )

    X_train, X_test, y_train, y_test = (
        __import__(
            "sklearn.model_selection",
            fromlist=["train_test_split"],
        ).train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y,
        )
    )

    print(
        f"Training data shape: {X_train.shape}"
    )

    print(
        f"Test data shape: {X_test.shape}"
    )

    model = build_accepted_model()

    print(
        "\nTraining the accepted Random Forest "
        "configuration..."
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    metrics = evaluate_model(
        y_test,
        predictions,
    )

    print(
        f"\nAccuracy: {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall: {metrics['recall']:.4f}"
    )

    print(
        f"F1: {metrics['f1']:.4f}"
    )

    if not metrics_match_expected(
        metrics
    ):
        raise RuntimeError(
            "Recreated model metrics do not match "
            "the accepted Chapter 8 evaluation."
        )

    print(
        "\nChapter 8 evaluation results reproduced "
        "successfully."
    )

    existing_version = find_existing_version(
        client,
        config.mlflow_model_name,
    )

    if existing_version is not None:
        print(
            "\nModel version already registered:"
        )
        print(
            f"Version: {existing_version.version}"
        )
        print(
            f"Run ID: {existing_version.run_id}"
        )

        return

    print(
        "\nLogging the deployable model to MLflow..."
    )

    with mlflow.start_run(
        run_name=(
            "chapter9-register-accepted-model"
        )
    ) as run:

        mlflow.set_tags(
            {
                "chapter": "9",
                "model_lifecycle": (
                    "registry-registration"
                ),
                "model_family": (
                    "random_forest"
                ),
                "model_role": "accepted",
                "validation_status": "approved",
                "source_evaluation_run_id": (
                    ACCEPTED_EVALUATION_RUN_ID
                ),
                "registration_status": (
                    "registered"
                ),
            }
        )

        mlflow.log_params(
            {
                "model_name": (
                    "RandomForestClassifier"
                ),
                "n_estimators": (
                    ACCEPTED_MODEL_PARAMETERS[
                        "n_estimators"
                    ]
                ),
                "min_samples_leaf": (
                    ACCEPTED_MODEL_PARAMETERS[
                        "min_samples_leaf"
                    ]
                ),
                "max_depth": (
                    ACCEPTED_MODEL_PARAMETERS[
                        "max_depth"
                    ]
                ),
                "random_state": (
                    config.random_state
                ),
                "class_weight": (
                    config.class_weight
                ),
                "source_evaluation_run_id": (
                    ACCEPTED_EVALUATION_RUN_ID
                ),
            }
        )

        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

        mlflow.sklearn.log_model(
            model,
            name=(
                config.mlflow_model_artifact_path
            ),
        )

        run_id = run.info.run_id

        model_uri = (
            f"runs:/{run_id}/"
            f"{config.mlflow_model_artifact_path}"
        )

        print(
            f"\nModel logging run: {run_id}"
        )

        print(
            f"Model URI: {model_uri}"
        )

        print(
            "\nRegistering model version..."
        )

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=config.mlflow_model_name,
        )

        print(
            f"Registered version: "
            f"{model_version.version}"
        )

        client.set_model_version_tag(
            name=config.mlflow_model_name,
            version=model_version.version,
            key="chapter",
            value="9",
        )

        client.set_model_version_tag(
            name=config.mlflow_model_name,
            version=model_version.version,
            key="model_role",
            value="accepted",
        )

        client.set_model_version_tag(
            name=config.mlflow_model_name,
            version=model_version.version,
            key="validation_status",
            value="approved",
        )

        client.set_model_version_tag(
            name=config.mlflow_model_name,
            version=model_version.version,
            key="source_evaluation_run_id",
            value=(
                ACCEPTED_EVALUATION_RUN_ID
            ),
        )

        client.set_model_version_tag(
            name=config.mlflow_model_name,
            version=model_version.version,
            key="registration_run_id",
            value=run_id,
        )

        client.set_registered_model_alias(
            name=config.mlflow_model_name,
            alias="development",
            version=model_version.version,
        )

        print(
            "\nModel registration completed."
        )

        print(
            f"Model: "
            f"{config.mlflow_model_name}"
        )

        print(
            f"Version: "
            f"{model_version.version}"
        )

        print(
            "Alias: development"
        )


if __name__ == "__main__":
    main()