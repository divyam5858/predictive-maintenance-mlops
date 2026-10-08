import os

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Central configuration for the predictive maintenance ML pipeline."""

    # Data splitting
    test_size: float = 0.2
    random_state: int = 42

    # Random Forest configuration
    n_estimators: int = int(
        os.getenv(
            "PM_N_ESTIMATORS",
            "100",
        )
    )
    class_weight: str = "balanced"

    # MLflow configuration
    mlflow_tracking_uri: str = (
        "http://127.0.0.1:5000"
    )
    mlflow_experiment_name: str = (
        "Predictive Maintenance"
    )

    # MLflow Model Registry configuration
    mlflow_model_name: str = (
        "predictive-maintenance-model"
    )
    mlflow_model_artifact_path: str = (
        "model"
    )