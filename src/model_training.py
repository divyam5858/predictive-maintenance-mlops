from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def create_random_forest(
    n_estimators: int = 100,
    random_state: int = 42,
    class_weight: str = "balanced",
) -> RandomForestClassifier:
    """
    Create the Random Forest baseline model.
    """

    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
    )


def create_xgboost(
    n_estimators: int = 100,
    random_state: int = 42,
) -> XGBClassifier:
    """
    Create the XGBoost classifier for predictive maintenance.
    """

    return XGBClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        eval_metric="logloss",
    )


def create_lightgbm(
    n_estimators: int = 100,
    random_state: int = 42,
) -> LGBMClassifier:
    """
    Create the LightGBM classifier for predictive maintenance.
    """

    return LGBMClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        verbosity=-1,
    )


def create_candidate_models(
    n_estimators: int = 100,
    random_state: int = 42,
) -> dict:
    """
    Create all candidate models used in the Chapter 8 benchmark.
    """

    return {
        "random_forest": create_random_forest(
            n_estimators=n_estimators,
            random_state=random_state,
        ),
        "xgboost": create_xgboost(
            n_estimators=n_estimators,
            random_state=random_state,
        ),
        "lightgbm": create_lightgbm(
            n_estimators=n_estimators,
            random_state=random_state,
        ),
    }
