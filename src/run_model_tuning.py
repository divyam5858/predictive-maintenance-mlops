from src.data_cleaning import clean_sensor_data
from src.data_ingestion import load_sensor_data
from src.data_validation import validate_data
from src.feature_pipeline import prepare_features
from src.gx_validation import validate_with_great_expectations
from src.model_training import (
    create_lightgbm,
    create_random_forest,
    create_xgboost,
)
from src.model_tuning import tune_model


def main():
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

    models_and_grids = {
        "random_forest": (
            create_random_forest(),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 10],
                "min_samples_leaf": [1, 2],
            },
        ),
        "xgboost": (
            create_xgboost(),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
        ),
        "lightgbm": (
            create_lightgbm(),
            {
                "n_estimators": [100, 200],
                "num_leaves": [15, 31],
                "learning_rate": [0.05, 0.1],
            },
        ),
    }

    tuning_results = {}

    print(
        "\nStarting hyperparameter optimization..."
    )

    print(
        "=============================================="
    )

    for model_name, (
        model,
        parameter_grid,
    ) in models_and_grids.items():

        print(
            f"\nTuning {model_name}..."
        )

        result = tune_model(
            model=model,
            parameter_grid=parameter_grid,
            X=X,
            y=y,
            cv=5,
            random_state=42,
        )

        tuning_results[model_name] = result

        print(
            f"Best F1: "
            f"{result['best_score']:.4f}"
        )

        print(
            "Best parameters:"
        )

        for parameter, value in (
            result["best_parameters"].items()
        ):
            print(
                f"  {parameter}: {value}"
            )

    print(
        "\nHyperparameter optimization completed."
    )


if __name__ == "__main__":
    main()
