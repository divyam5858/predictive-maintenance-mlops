import pandas as pd

from sklearn.pipeline import Pipeline
from zenml import step


@step(
    name="run_production_inference",
    enable_artifact_metadata=True,
)
def run_production_inference(
    model: Pipeline,
    features: pd.DataFrame,
) -> pd.Series:
    """
    Generate predictions using the production-registered model.
    """

    predictions = model.predict(features)

    predictions_series = pd.Series(
        predictions,
        index=features.index,
        name="prediction",
    )

    print(
        f"Production inference samples: "
        f"{len(predictions_series)}"
    )

    print(
        "Production prediction distribution:"
    )
    print(
        predictions_series.value_counts()
    )

    return predictions_series