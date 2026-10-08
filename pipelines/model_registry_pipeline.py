from zenml import pipeline

from src.registry_model_step import load_production_model


@pipeline
def model_registry_pipeline(
    environment: str = "production",
):
    """
    ZenML pipeline for loading a model from
    the MLflow Model Registry.
    """

    load_production_model(
        environment=environment,
    )