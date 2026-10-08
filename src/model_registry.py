import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from configs.pipeline_config import PipelineConfig


def get_mlflow_client() -> MlflowClient:
    """Create an MLflow client using the centralized configuration."""

    config = PipelineConfig()

    mlflow.set_tracking_uri(
        config.mlflow_tracking_uri
    )

    return MlflowClient(
        tracking_uri=config.mlflow_tracking_uri
    )


def get_registered_model(
    model_name: str | None = None,
):
    """Return a registered model by name."""

    config = PipelineConfig()

    name = (
        model_name
        if model_name is not None
        else config.mlflow_model_name
    )

    client = get_mlflow_client()

    try:
        return client.get_registered_model(name)

    except MlflowException as exc:
        raise ValueError(
            f"Registered model '{name}' was not found."
        ) from exc


def get_model_versions(
    model_name: str | None = None,
):
    """Return all versions of a registered model."""

    config = PipelineConfig()

    name = (
        model_name
        if model_name is not None
        else config.mlflow_model_name
    )

    client = get_mlflow_client()

    versions = client.search_model_versions(
        f"name='{name}'"
    )

    return sorted(
        versions,
        key=lambda version: int(version.version),
    )


def register_model_from_run(
    run_id: str,
    model_name: str | None = None,
):
    """
    Register an MLflow model artifact from an existing run.

    The trained model is expected to be stored under
    the MLflow artifact path configured for the project.
    """

    config = PipelineConfig()

    name = (
        model_name
        if model_name is not None
        else config.mlflow_model_name
    )

    model_uri = (
        f"runs:/{run_id}/"
        f"{config.mlflow_model_artifact_path}"
    )

    client = get_mlflow_client()

    try:
        client.get_registered_model(name)

    except MlflowException:
        client.create_registered_model(name)

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=name,
    )

    return model_version


def set_model_tags(
    model_name: str,
    version: str,
    tags: dict[str, str],
) -> None:
    """Attach governance and lifecycle metadata to a model version."""

    client = get_mlflow_client()

    for key, value in tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key=key,
            value=value,
        )


def set_model_alias(
    model_name: str,
    alias: str,
    version: str,
) -> None:
    """Assign an MLflow alias to a model version."""

    client = get_mlflow_client()

    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version,
    )


def get_model_version_by_alias(
    model_name: str,
    alias: str,
):
    """Retrieve a model version using its lifecycle alias."""

    client = get_mlflow_client()

    try:
        return client.get_model_version_by_alias(
            name=model_name,
            alias=alias,
        )

    except MlflowException as exc:
        raise ValueError(
            f"No model version is assigned to "
            f"alias '{alias}'."
        ) from exc