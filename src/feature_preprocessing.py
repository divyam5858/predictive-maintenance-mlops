import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


CATEGORICAL_FEATURES = [
    "machine_id",
]


NUMERICAL_FEATURES = [
    "ambient_temperature",
    "humidity",
    "temperature",
    "vibration",
    "pressure",
    "rotational_speed",
    "torque",
    "operating_hours",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "temperature_difference",
    "mechanical_load",
    "operating_hours_squared",
]


def create_feature_preprocessor() -> ColumnTransformer:
    """
    Create the reusable feature transformation pipeline.

    Numerical features are standardized using StandardScaler.
    Machine ID is treated as a categorical feature and encoded
    using OneHotEncoder.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def fit_feature_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """
    Fit the feature preprocessor using training data only.
    """

    preprocessor = create_feature_preprocessor()

    preprocessor.fit(X_train)

    return preprocessor


def transform_features(
    preprocessor: ColumnTransformer,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform data using an already-fitted preprocessor.

    The same fitted transformation should be reused for
    validation/test and production inference data.
    """

    transformed = preprocessor.transform(data)

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    return pd.DataFrame(
        transformed,
        columns=feature_names,
        index=data.index,
    )