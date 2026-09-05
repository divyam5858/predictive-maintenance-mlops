from zenml import pipeline

from src.ingestion_step import ingest_data
from src.cleaning_step import clean_data
from src.validation_step import validate_sensor_data
from src.feature_engineering_step import (
    engineer_and_select_features,
)
from src.training_step import train_model
from src.inference_step import predict_failure


@pipeline
def predictive_maintenance_pipeline():
    """
    End-to-end predictive maintenance pipeline.

    The pipeline ingests, cleans, validates, engineers,
    selects, preprocesses, trains, and predicts.
    """

    data = ingest_data()

    cleaned_data = clean_data(data)

    validated_data = validate_sensor_data(
        cleaned_data
    )

    X, y = engineer_and_select_features(
        validated_data
    )

    (
        model,
        preprocessor,
        X_test,
        y_test,
        metrics,
    ) = train_model(
        X,
        y,
    )

    predict_failure(
        model,
        X_test,
        y_test,
    )