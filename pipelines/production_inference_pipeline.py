from zenml import pipeline

from src.cleaning_step import clean_data
from src.feature_engineering_step import (
    engineer_and_select_features,
)
from src.ingestion_step import ingest_data
from src.production_inference_step import (
    run_production_inference,
)
from src.registry_model_step import load_production_model
from src.validation_step import validate_sensor_data


@pipeline
def production_inference_pipeline(
    environment: str = "production",
):
    """
    Run inference using a model retrieved from
    the MLflow Model Registry.
    """

    data = ingest_data()

    cleaned_data = clean_data(
        data
    )

    validated_data = validate_sensor_data(
        cleaned_data
    )

    features, _ = engineer_and_select_features(
        validated_data
    )

    model = load_production_model(
        environment=environment
    )

    run_production_inference(
        model=model,
        features=features,
    )