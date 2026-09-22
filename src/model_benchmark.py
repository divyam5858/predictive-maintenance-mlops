import time

import pandas as pd
from sklearn.model_selection import train_test_split

from configs.pipeline_config import PipelineConfig
from src.feature_pipeline import (
    fit_and_transform_training_features,
    transform_model_features,
)
from src.model_evaluation import evaluate_model
from src.model_training import create_candidate_models


def prepare_train_test_data(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Create the reproducible train/test split and transform
    the features using a preprocessor fitted only on training data.
    """

    config = PipelineConfig()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    X_train_transformed, preprocessor = (
        fit_and_transform_training_features(
            X_train
        )
    )

    X_test_transformed = transform_model_features(
        preprocessor,
        X_test,
    )

    return (
        X_train_transformed,
        X_test_transformed,
        y_train,
        y_test,
        preprocessor,
    )


def benchmark_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Train and evaluate all baseline candidate models
    using the same prepared training and test data.
    """

    config = PipelineConfig()

    models = create_candidate_models(
        n_estimators=config.n_estimators,
        random_state=config.random_state,
    )

    results = []

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name}..."
        )

        start_time = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        predictions = model.predict(
            X_test
        )

        metrics = evaluate_model(
            y_test,
            predictions,
        )

        result = {
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "training_time_seconds": training_time,
        }

        results.append(result)

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
            f"Training Time: "
            f"{training_time:.4f} seconds"
        )

    return pd.DataFrame(results)
