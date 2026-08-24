from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Central configuration for the predictive maintenance ML pipeline."""

    # Data splitting
    test_size: float = 0.2
    random_state: int = 42

    # Random Forest configuration
    n_estimators: int = 100
    class_weight: str = "balanced"

    # MLflow configuration
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment_name: str = "Predictive Maintenance"