from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.model_registry import get_mlflow_client


ENVIRONMENT_ALIASES = (
    "development",
    "staging",
    "production",
)


def validate_environment(environment: str) -> None:
    """Validate that the environment is supported."""

    if environment not in ENVIRONMENT_ALIASES:
        raise ValueError(
            f"Unsupported environment '{environment}'. "
            f"Choose one of: {', '.join(ENVIRONMENT_ALIASES)}."
        )


def get_model_version(
    model_name: str,
    version: str,
):
    """Retrieve a specific registered model version."""

    client: MlflowClient = get_mlflow_client()

    try:
        return client.get_model_version(
            name=model_name,
            version=version,
        )
    except MlflowException as exc:
        raise ValueError(
            f"Model '{model_name}' version '{version}' "
            "was not found."
        ) from exc


def promote_model(
    model_name: str,
    version: str,
    environment: str,
) -> None:
    """
    Promote a registered model version to an environment.

    MLflow aliases are used to represent deployment environments.
    """

    validate_environment(environment)

    model_version = get_model_version(
        model_name=model_name,
        version=version,
    )

    client = get_mlflow_client()

    validation_status = model_version.tags.get(
        "validation_status"
    )

    if validation_status != "approved":
        raise ValueError(
            f"Model version {version} cannot be promoted. "
            f"Expected validation_status='approved', "
            f"found '{validation_status}'."
        )

    client.set_registered_model_alias(
        name=model_name,
        alias=environment,
        version=version,
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="deployment_environment",
        value=environment,
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="promotion_status",
        value="promoted",
    )

    print(
        f"Model '{model_name}' version {version} "
        f"promoted to '{environment}'."
    )


def get_model_version_by_environment(
    model_name: str,
    environment: str,
):
    """Retrieve the model version currently assigned to an environment."""

    validate_environment(environment)

    client = get_mlflow_client()

    try:
        return client.get_model_version_by_alias(
            name=model_name,
            alias=environment,
        )
    except MlflowException as exc:
        raise ValueError(
            f"No model is currently assigned to "
            f"environment '{environment}'."
        ) from exc


def print_environment_status(
    model_name: str,
) -> None:
    """Display the model version assigned to each environment."""

    print(f"\nModel: {model_name}")
    print("-" * 60)

    for environment in ENVIRONMENT_ALIASES:
        try:
            model_version = get_model_version_by_environment(
                model_name=model_name,
                environment=environment,
            )

            print(
                f"{environment.capitalize():12} "
                f"Version {model_version.version}"
            )

        except ValueError:
            print(
                f"{environment.capitalize():12} "
                "Not assigned"
            )