from sklearn.pipeline import Pipeline
from zenml import step

from src.registry_model_loader import load_registered_model


@step(
    name="load_registered_model",
    enable_artifact_metadata=True,
)
def load_production_model(
    environment: str = "production",
) -> Pipeline:
    """
    Load a model from the MLflow Model Registry.

    The model is resolved through an MLflow environment
    alias such as development, staging, or production.
    """

    model = load_registered_model(
        environment=environment,
    )

    print(
        f"Loaded registered model from "
        f"'{environment}' environment."
    )

    print(
        f"Model type: {type(model)}"
    )

    return model