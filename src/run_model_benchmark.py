from src.data_cleaning import clean_sensor_data
from src.data_ingestion import load_sensor_data
from src.data_validation import validate_data
from src.feature_pipeline import prepare_features
from src.gx_validation import validate_with_great_expectations
from src.model_benchmark import (
    benchmark_models,
    prepare_train_test_data,
)


def main():
    print("Loading raw sensor data...")

    data = load_sensor_data()

    print(
        f"Raw dataset shape: {data.shape}"
    )

    print("Cleaning sensor data...")

    cleaned_data = clean_sensor_data(
        data
    )

    print(
        f"Cleaned dataset shape: "
        f"{cleaned_data.shape}"
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
        f"Feature matrix shape: {X.shape}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    print("Preparing train/test data...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        _,
    ) = prepare_train_test_data(
        X,
        y,
    )

    print(
        f"Training data shape: "
        f"{X_train.shape}"
    )

    print(
        f"Test data shape: "
        f"{X_test.shape}"
    )

    print("\nRunning baseline model benchmark...")

    results = benchmark_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nBaseline Benchmark Results")
    print("==========================")

    print(
        results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
