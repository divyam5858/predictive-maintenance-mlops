import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    y_true: pd.Series,
    predictions,
) -> dict[str, float]:
    """
    Calculate classification metrics for a trained model.

    The same evaluation logic is used for all candidate
    models in the Predictive Maintenance System.
    """

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }

    return metrics


def generate_classification_report(
    y_true: pd.Series,
    predictions,
) -> str:
    """
    Generate a detailed classification report.
    """

    return classification_report(
        y_true,
        predictions,
        digits=4,
        zero_division=0,
    )


def generate_confusion_matrix(
    y_true: pd.Series,
    predictions,
) -> pd.DataFrame:
    """
    Generate a labeled confusion matrix as a DataFrame.
    """

    values = confusion_matrix(
        y_true,
        predictions,
    )

    return pd.DataFrame(
        values,
        index=[
            "actual_0",
            "actual_1",
        ],
        columns=[
            "predicted_0",
            "predicted_1",
        ],
    )
