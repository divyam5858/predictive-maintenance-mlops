from zenml import pipeline

from src.ingestion_step import ingest_data
from src.preprocessing_step import preprocess_data
from src.training_step import train_model
from src.inference_step import predict_failure


@pipeline
def predictive_maintenance_pipeline():
    """
    End-to-end predictive maintenance machine learning pipeline.

    Workflow:
        1. Ingest raw sensor data
        2. Preprocess features and target
        3. Train and evaluate the model
        4. Generate predictions on unseen data
    """

    # Step 1: Ingest raw sensor data.
    data = ingest_data()

    # Step 2: Preprocess data into features and target.
    X, y = preprocess_data(data)

    # Step 3: Train the model and prepare test data.
    model, X_test, y_test, metrics = train_model(X, y)

    # Step 4: Generate predictions on unseen test data.
    predict_failure(model, X_test, y_test)