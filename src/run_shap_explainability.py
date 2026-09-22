from sklearn.model_selection import train_test_split

from configs.pipeline_config import PipelineConfig
from src.data_cleaning import clean_sensor_data
from src.data_ingestion import load_sensor_data
from src.data_validation import validate_data
from src.feature_pipeline import (
    fit_and_transform_training_features,
    prepare_features,
    transform_model_features,
)
from src.gx_validation import validate_with_great_expectations
from src.model_explainability import (
    explain_tree_model,
    save_shap_artifacts,
)
from src.model_training import create_random_forest


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

    print(
        "Fitting feature preprocessing..."
    )

    X_train_transformed, preprocessor = (
        fit_and_transform_training_features(
            X_train
        )
    )

    X_test_transformed = (
        transform_model_features(
            preprocessor,
            X_test,
        )
    )

    print(
        f"Transformed test shape: "
        f"{X_test_transformed.shape}"
    )

    print(
        "Training Random Forest..."
    )

    model = create_random_forest(
        n_estimators=200,
        random_state=config.random_state,
    )

    model.fit(
        X_train_transformed,
        y_train,
    )

    print(
        "Calculating SHAP values..."
    )

    # Explain a reproducible subset of 500 test
    # samples to keep the explanation artifact
    # lightweight.
    X_explain = X_test_transformed.iloc[
        :500
    ]

    shap_values, feature_importance = (
        explain_tree_model(
            model,
            X_explain,
        )
    )

    print(
        f"SHAP values shape: "
        f"{shap_values.shape}"
    )

    print(
        f"Feature importance shape: "
        f"{feature_importance.shape}"
    )

    print(
        "\nTop 10 SHAP features:"
    )

    print(
        feature_importance.head(10).to_string(
            index=False
        )
    )

    artifact_paths = save_shap_artifacts(
        shap_values,
        feature_importance,
        X_explain.columns,
    )

    print(
        "\nSHAP artifacts saved:"
    )

    for artifact_name, path in (
        artifact_paths.items()
    ):
        print(
            f"{artifact_name}: {path}"
        )


if __name__ == "__main__":
    main()
