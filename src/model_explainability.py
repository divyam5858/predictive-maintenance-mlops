from pathlib import Path

import pandas as pd
import shap


DEFAULT_EXPLAINABILITY_DIR = Path(
    "artifacts/explainability"
)


def create_tree_explainer(
    model,
) -> shap.TreeExplainer:
    """
    Create a SHAP TreeExplainer for a tree-based model.
    """

    return shap.TreeExplainer(
        model
    )


def calculate_shap_values(
    model,
    X: pd.DataFrame,
):
    """
    Calculate SHAP values for a tree-based model.

    For binary classification, the SHAP values corresponding
    to the positive failure class are returned.
    """

    explainer = create_tree_explainer(
        model
    )

    shap_values = explainer.shap_values(
        X
    )

    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            return shap_values[1]

        return shap_values[0]

    if hasattr(shap_values, "ndim"):
        if shap_values.ndim == 3:
            return shap_values[:, :, 1]

        return shap_values

    raise TypeError(
        "Unsupported SHAP value format."
    )


def calculate_shap_feature_importance(
    shap_values,
    feature_names,
) -> pd.DataFrame:
    """
    Calculate global feature importance using the
    mean absolute SHAP value for each feature.
    """

    importance = pd.DataFrame(
        {
            "feature": list(feature_names),
            "mean_absolute_shap": (
                abs(shap_values).mean(axis=0)
            ),
        }
    )

    return importance.sort_values(
        "mean_absolute_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )


def save_shap_artifacts(
    shap_values,
    feature_importance: pd.DataFrame,
    feature_names,
    output_dir: Path = DEFAULT_EXPLAINABILITY_DIR,
) -> dict[str, Path]:
    """
    Save SHAP values and global feature importance
    as reusable explainability artifacts.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shap_values_path = (
        output_dir /
        "shap_values.csv"
    )

    shap_values_df = pd.DataFrame(
        shap_values,
        columns=feature_names,
    )

    shap_values_df.to_csv(
        shap_values_path,
        index=False,
    )

    feature_importance_path = (
        output_dir /
        "shap_feature_importance.csv"
    )

    feature_importance.to_csv(
        feature_importance_path,
        index=False,
    )

    return {
        "shap_values": shap_values_path,
        "feature_importance": (
            feature_importance_path
        ),
    }


def explain_tree_model(
    model,
    X: pd.DataFrame,
) -> tuple:
    """
    Calculate SHAP values and global feature importance
    for a tree-based model.

    Returns:
        shap_values:
            SHAP values for the positive failure class.

        feature_importance:
            DataFrame containing features ranked by mean
            absolute SHAP value.
    """

    shap_values = calculate_shap_values(
        model,
        X,
    )

    feature_importance = (
        calculate_shap_feature_importance(
            shap_values,
            X.columns,
        )
    )

    return (
        shap_values,
        feature_importance,
    )
