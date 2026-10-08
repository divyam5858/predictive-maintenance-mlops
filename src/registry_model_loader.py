import mlflow
import mlflow.sklearn

from configs.pipeline_config import PipelineConfig


def load_registered_model(
    environment: str = "production",
):
    """
    Load a registered Scikit-Learn model using an MLflow alias.

    Examples:
        development
        staging
        production
    """

    config = PipelineConfig()

    supported_environments = {
        "development",
        "staging",
        "production",
    }

    if environment not in supported_environments:
        raise ValueError(
            f"Unsupported environment '{environment}'. "
            f"Choose one of: "
            f"{', '.join(sorted(supported_environments))}."
        )

    mlflow.set_tracking_uri(
        config.mlflow_tracking_uri
    )

    model_uri = (
        f"models:/{config.mlflow_model_name}"
        f"@{environment}"
    )

    print(
        f"Loading model from MLflow Registry: "
        f"{model_uri}"
    )

    model = mlflow.sklearn.load_model(
        model_uri
    )

    print(
        f"Loaded model type: {type(model)}"
    )

    return model


def get_registered_model_uri(
    environment: str = "production",
) -> str:
    """Return the MLflow model URI for an environment."""

    config = PipelineConfig()

    supported_environments = {
        "development",
        "staging",
        "production",
    }

    if environment not in supported_environments:
        raise ValueError(
            f"Unsupported environment '{environment}'."
        )

    return (
        f"models:/{config.mlflow_model_name}"
        f"@{environment}"
    )