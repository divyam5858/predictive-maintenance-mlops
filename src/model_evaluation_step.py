import mlflow
import pandas as pd

from zenml import step

from configs.pipeline_config import PipelineConfig
from src.model_acceptance import (
    DEFAULT_ACCEPTANCE_CRITERIA,
    evaluate_model_acceptance,
)
from src.model_evaluation import (
    evaluate_model,
    generate_classification_report,
    generate_confusion_matrix,
)


@step(
    name="evaluate_predictive_maintenance_model",
    enable_artifact_metadata=True,
    enable_cache=False,
)
def evaluate_trained_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Evaluate the trained predictive-maintenance model
    inside the ZenML pipeline.

    Evaluation metrics are recorded in MLflow and the
    model is checked against the Chapter 8 acceptance
    criteria.
    """

    config = PipelineConfig()

    print(
        "Evaluating trained model..."
    )

    predictions = model.predict(
        X_test
    )

    metrics = evaluate_model(
        y_test,
        predictions,
    )

    acceptance = evaluate_model_acceptance(
        metrics,
        DEFAULT_ACCEPTANCE_CRITERIA,
    )

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

    print(
        f"Acceptance status: "
        f"{acceptance['accepted']}"
    )

    print("\nConfusion Matrix:")

    print(
        generate_confusion_matrix(
            y_test,
            predictions,
        )
    )

    print("\nClassification Report:")

    print(
        generate_classification_report(
            y_test,
            predictions,
        )
    )

    # Configure the existing MLflow tracking server.
    mlflow.set_tracking_uri(
        config.mlflow_tracking_uri
    )

    mlflow.set_experiment(
        config.mlflow_experiment_name
    )

    with mlflow.start_run(
        run_name="chapter8-zenml-evaluation"
    ):

        mlflow.set_tags(
            {
                "chapter": "8",
                "evaluation_stage": "zenml",
                "task": (
                    "predictive-maintenance-"
                    "classification"
                ),
                "acceptance_status": str(
                    acceptance["accepted"]
                ),
            }
        )

        for metric_name, value in metrics.items():
            mlflow.log_metric(
                f"test_{metric_name}",
                value,
            )

        mlflow.log_metric(
            "model_accepted",
            int(
                acceptance["accepted"]
            ),
        )

        mlflow.log_param(
            "minimum_recall",
            DEFAULT_ACCEPTANCE_CRITERIA[
                "minimum_recall"
            ],
        )

        mlflow.log_param(
            "minimum_precision",
            DEFAULT_ACCEPTANCE_CRITERIA[
                "minimum_precision"
            ],
        )

        mlflow.log_param(
            "minimum_f1",
            DEFAULT_ACCEPTANCE_CRITERIA[
                "minimum_f1"
            ],
        )

    return metrics
