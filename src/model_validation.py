import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.feature_preprocessing import (
    create_feature_preprocessor,
)
from src.model_evaluation import evaluate_model


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Perform leakage-safe stratified cross-validation.

    A fresh feature preprocessor is fitted inside every fold,
    using only that fold's training partition.
    """

    cross_validator = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_metrics = []

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cross_validator.split(X, y),
        start=1,
    ):
        fold_model = clone(model)

        fold_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_feature_preprocessor(),
                ),
                (
                    "model",
                    fold_model,
                ),
            ]
        )

        X_fold_train = X.iloc[train_indices]
        X_fold_validation = X.iloc[
            validation_indices
        ]

        y_fold_train = y.iloc[train_indices]
        y_fold_validation = y.iloc[
            validation_indices
        ]

        fold_pipeline.fit(
            X_fold_train,
            y_fold_train,
        )

        predictions = fold_pipeline.predict(
            X_fold_validation
        )

        metrics = evaluate_model(
            y_fold_validation,
            predictions,
        )

        fold_metrics.append(
            {
                "fold": fold_number,
                **metrics,
            }
        )

    fold_results = pd.DataFrame(
        fold_metrics
    )

    summary = {
        "accuracy_mean": fold_results[
            "accuracy"
        ].mean(),
        "accuracy_std": fold_results[
            "accuracy"
        ].std(),
        "precision_mean": fold_results[
            "precision"
        ].mean(),
        "precision_std": fold_results[
            "precision"
        ].std(),
        "recall_mean": fold_results[
            "recall"
        ].mean(),
        "recall_std": fold_results[
            "recall"
        ].std(),
        "f1_mean": fold_results[
            "f1"
        ].mean(),
        "f1_std": fold_results[
            "f1"
        ].std(),
    }

    return {
        "fold_results": fold_results,
        "summary": summary,
    }
