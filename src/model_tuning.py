import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.feature_preprocessing import (
    create_feature_preprocessor,
)


def tune_model(
    model,
    parameter_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Perform leakage-safe hyperparameter optimization.

    The feature preprocessor is included inside the
    GridSearchCV pipeline so that preprocessing is fitted
    independently within every cross-validation split.

    StratifiedKFold is used to preserve the class distribution
    across folds.

    F1-score is used as the optimization metric.
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_feature_preprocessor(),
            ),
            (
                "model",
                clone(model),
            ),
        ]
    )

    prefixed_parameters = {
        f"model__{parameter}": values
        for parameter, values in parameter_grid.items()
    }

    cross_validator = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=prefixed_parameters,
        scoring="f1",
        cv=cross_validator,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    search.fit(
        X,
        y,
    )

    return {
        "search": search,
        "best_model": search.best_estimator_,
        "best_parameters": search.best_params_,
        "best_score": search.best_score_,
    }
