import json
import os
import tempfile

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from configs.pipeline_config import PipelineConfig
from src.data_cleaning import clean_sensor_data
from src.data_ingestion import load_sensor_data
from src.data_validation import validate_data
from src.feature_pipeline import prepare_features
from src.gx_validation import validate_with_great_expectations
from src.model_acceptance import (
    DEFAULT_ACCEPTANCE_CRITERIA,
    evaluate_model_acceptance,
)
from src.model_evaluation import (
    evaluate_model,
    generate_classification_report,
    generate_confusion_matrix,
)
from src.model_training import (
    create_lightgbm,
    create_random_forest,
    create_xgboost,
)
from src.model_tuning import tune_model


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

    models_and_grids = {
        "random_forest": (
            create_random_forest(
                n_estimators=config.n_estimators,
                random_state=config.random_state,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 10],
                "min_samples_leaf": [1, 2],
            },
        ),
        "xgboost": (
            create_xgboost(
                n_estimators=config.n_estimators,
                random_state=config.random_state,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
        ),
        "lightgbm": (
            create_lightgbm(
                n_estimators=config.n_estimators,
                random_state=config.random_state,
            ),
            {
                "n_estimators": [100, 200],
                "num_leaves": [15, 31],
                "learning_rate": [0.05, 0.1],
            },
        ),
    }

    final_results = []

    print(
        "\nTuning and evaluating candidate models..."
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

        tuning_result = tune_model(
            model=model,
            parameter_grid=parameter_grid,
            X=X_train,
            y=y_train,
            cv=5,
            random_state=config.random_state,
        )

        best_model = tuning_result[
            "best_model"
        ]

        best_score = tuning_result[
            "best_score"
        ]

        best_parameters = tuning_result[
            "best_parameters"
        ]

        print(
            f"Best CV F1: "
            f"{best_score:.4f}"
        )

        print(
            "Best parameters:"
        )

        for parameter, value in (
            best_parameters.items()
        ):
            print(
                f"  {parameter}: {value}"
            )

        print(
            "Evaluating on untouched test data..."
        )

        predictions = best_model.predict(
            X_test
        )

        metrics = evaluate_model(
            y_test,
            predictions,
        )

        acceptance = (
            evaluate_model_acceptance(
                metrics,
                DEFAULT_ACCEPTANCE_CRITERIA,
            )
        )

        print(
            f"Test Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Test Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Test Recall: "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"Test F1: "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"Acceptance status: "
            f"{acceptance['accepted']}"
        )

        confusion_matrix_df = (
            generate_confusion_matrix(
                y_test,
                predictions,
            )
        )

        classification_report_text = (
            generate_classification_report(
                y_test,
                predictions,
            )
        )

        print("\nConfusion Matrix:")

        print(
            confusion_matrix_df
        )

        print("\nClassification Report:")

        print(
            classification_report_text
        )

        # -------------------------------------------------
        # MLflow evaluation tracking
        # -------------------------------------------------

        mlflow.set_tracking_uri(
            config.mlflow_tracking_uri
        )

        mlflow.set_experiment(
            config.mlflow_experiment_name
        )

        with mlflow.start_run(
            run_name=(
                f"chapter8-{model_name}-tuned"
            )
        ):

            mlflow.set_tags(
                {
                    "chapter": "8",
                    "evaluation_stage": "tuned",
                    "task": (
                        "predictive-maintenance-"
                        "classification"
                    ),
                    "model_family": model_name,
                    "acceptance_status": str(
                        acceptance["accepted"]
                    ),
                }
            )

            # Training and evaluation configuration.
            mlflow.log_param(
                "model_name",
                model_name,
            )

            mlflow.log_param(
                "test_size",
                config.test_size,
            )

            mlflow.log_param(
                "random_state",
                config.random_state,
            )

            mlflow.log_param(
                "cv_folds",
                5,
            )

            mlflow.log_param(
                "optimization_metric",
                "f1",
            )

            # Log best hyperparameters discovered
            # by GridSearchCV.
            for parameter, value in (
                best_parameters.items()
            ):
                mlflow.log_param(
                    parameter,
                    str(value),
                )

            # Log cross-validation result.
            mlflow.log_metric(
                "cv_f1",
                best_score,
            )

            # Log untouched test-set metrics.
            for metric_name, value in (
                metrics.items()
            ):
                mlflow.log_metric(
                    f"test_{metric_name}",
                    value,
                )

            # Log acceptance criteria.
            mlflow.log_param(
                "minimum_recall",
                DEFAULT_ACCEPTANCE_CRITERIA[
                    "minimum_recall"
                ],
            )

            mlflow.log_param(
                "minimum_precision",
                DEFAULT_ACCEPTANCE_CRITERIA[
                    "minimum_precision"
                ],
            )

            mlflow.log_param(
                "minimum_f1",
                DEFAULT_ACCEPTANCE_CRITERIA[
                    "minimum_f1"
                ],
            )

            # Log acceptance results.
            mlflow.log_metric(
                "acceptance_recall_passed",
                int(
                    acceptance[
                        "recall_passed"
                    ]
                ),
            )

            mlflow.log_metric(
                "acceptance_precision_passed",
                int(
                    acceptance[
                        "precision_passed"
                    ]
                ),
            )

            mlflow.log_metric(
                "acceptance_f1_passed",
                int(
                    acceptance[
                        "f1_passed"
                    ]
                ),
            )

            mlflow.log_metric(
                "model_accepted",
                int(
                    acceptance[
                        "accepted"
                    ]
                ),
            )

            # Create temporary evaluation artifacts.
            with tempfile.TemporaryDirectory() as temp_dir:

                confusion_matrix_path = os.path.join(
                    temp_dir,
                    "confusion_matrix.csv",
                )

                confusion_matrix_df.to_csv(
                    confusion_matrix_path
                )

                classification_report_path = os.path.join(
                    temp_dir,
                    "classification_report.txt",
                )

                with open(
                    classification_report_path,
                    "w",
                    encoding="utf-8",
                ) as file:
                    file.write(
                        classification_report_text
                    )

                acceptance_path = os.path.join(
                    temp_dir,
                    "acceptance_result.json",
                )

                with open(
                    acceptance_path,
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        {
                            "model": model_name,
                            "criteria": (
                                DEFAULT_ACCEPTANCE_CRITERIA
                            ),
                            "results": acceptance,
                        },
                        file,
                        indent=4,
                    )

                mlflow.log_artifacts(
                    temp_dir,
                    artifact_path=(
                        "evaluation"
                    ),
                )

        final_results.append(
            {
                "model": model_name,
                "cv_f1": best_score,
                "test_accuracy": metrics[
                    "accuracy"
                ],
                "test_precision": metrics[
                    "precision"
                ],
                "test_recall": metrics[
                    "recall"
                ],
                "test_f1": metrics[
                    "f1"
                ],
                "accepted": acceptance[
                    "accepted"
                ],
            }
        )

    print(
        "\nFinal Tuned Model Results"
    )

    print(
        "=========================="
    )

    results_df = pd.DataFrame(
        final_results
    )

    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
