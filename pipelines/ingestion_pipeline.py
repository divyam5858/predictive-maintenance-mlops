from zenml import pipeline

from src.ingestion_step import ingest_data
from src.preprocessing_step import preprocess_data
from src.training_step import train_model
from src.inference_step import predict_failure


@pipeline
def ingestion_pipeline():

    # Step 1: Load raw data.
    data = ingest_data()

    # Step 2: Prepare features and target.
    X, y = preprocess_data(data)

    # Step 3: Train model and evaluate it.
    model, X_test, y_test, metrics = train_model(X, y)

    # Step 4: Generate predictions on unseen test data.
    predict_failure(model, X_test, y_test)