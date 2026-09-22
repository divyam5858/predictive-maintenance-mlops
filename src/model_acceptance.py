import pandas as pd


DEFAULT_ACCEPTANCE_CRITERIA = {
    "minimum_recall": 0.50,
    "minimum_precision": 0.40,
    "minimum_f1": 0.45,
}


def evaluate_model_acceptance(
    metrics: dict[str, float],
    criteria: dict[str, float] | None = None,
) -> dict:
    """
    Evaluate a model against explicit acceptance criteria.
    """

    if criteria is None:
        criteria = DEFAULT_ACCEPTANCE_CRITERIA

    recall_passed = (
        metrics["recall"]
        >= criteria["minimum_recall"]
    )

    precision_passed = (
        metrics["precision"]
        >= criteria["minimum_precision"]
    )

    f1_passed = (
        metrics["f1"]
        >= criteria["minimum_f1"]
    )

    accepted = (
        recall_passed
        and precision_passed
        and f1_passed
    )

    return {
        "accepted": accepted,
        "recall_passed": recall_passed,
        "precision_passed": precision_passed,
        "f1_passed": f1_passed,
    }


def compare_model_acceptance(
    results: pd.DataFrame,
    criteria: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate acceptance criteria for multiple models.

    The input DataFrame must contain:
        model
        test_precision
        test_recall
        test_f1
    """

    if criteria is None:
        criteria = DEFAULT_ACCEPTANCE_CRITERIA

    records = []

    for _, row in results.iterrows():
        metrics = {
            "precision": row["test_precision"],
            "recall": row["test_recall"],
            "f1": row["test_f1"],
        }

        acceptance = evaluate_model_acceptance(
            metrics,
            criteria,
        )

        records.append(
            {
                "model": row["model"],
                "test_precision": row[
                    "test_precision"
                ],
                "test_recall": row[
                    "test_recall"
                ],
                "test_f1": row[
                    "test_f1"
                ],
                "accepted": acceptance[
                    "accepted"
                ],
            }
        )

    return pd.DataFrame(records)
