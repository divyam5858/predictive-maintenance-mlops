from sklearn.model_selection import train_test_split

from configs.pipeline_config import PipelineConfig
from src.data_cleaning import clean_sensor_data
from src.data_ingestion import load_sensor_data
from src.data_validation import validate_data
from src.feature_pipeline import prepare_features
from src.gx_validation import validate_with_great_expectations
from src.model_training import create_candidate_models
from src.model_validation import cross_validate_model


def main():
    config = PipelineConfig()

    print("Loading raw sensor data...")

    data = load_sensor_data()

    cleaned_data = clean_sensor_data(
        data
    )

    print("Running Pandas/NumPy validation...")

    validate_data(
        cleaned_data
    )

    print(
        "Pandas/NumPy validation passed."
    )

    print(
        "Running Great Expectations validation..."
    )

    validate_with_great_expectations(
        cleaned_data
    )

    print(
        "Great Expectations validation passed."
    )

    print("Preparing model features...")

    X, y = prepare_features(
        cleaned_data
    )

    print(
        f"Selected feature shape: {X.shape}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    # Create the same reproducible train/test split
    # used by the Chapter 7 training workflow.
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y,
        )
    )

    print(
        f"Training data shape: {X_train.shape}"
    )

    print(
        f"Test data shape: {X_test.shape}"
    )

    models = create_candidate_models(
        n_estimators=config.n_estimators,
        random_state=config.random_state,
    )

    print(
        "\nRunning leakage-safe 5-fold "
        "stratified cross-validation..."
    )

    print(
        "=============================================="
    )

    for model_name, model in models.items():

        print(
            f"\nModel: {model_name}"
        )

        results = cross_validate_model(
            model,
            X_train,
            y_train,
            n_splits=5,
            random_state=config.random_state,
        )

        summary = results["summary"]

        print(
            f"Accuracy:  "
            f"{summary['accuracy_mean']:.4f} "
            f"+/- {summary['accuracy_std']:.4f}"
        )

        print(
            f"Precision: "
            f"{summary['precision_mean']:.4f} "
            f"+/- {summary['precision_std']:.4f}"
        )

        print(
            f"Recall:    "
            f"{summary['recall_mean']:.4f} "
            f"+/- {summary['recall_std']:.4f}"
        )

        print(
            f"F1:        "
            f"{summary['f1_mean']:.4f} "
            f"+/- {summary['f1_std']:.4f}"
        )

        print("\nFold results:")

        print(
            results["fold_results"].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
