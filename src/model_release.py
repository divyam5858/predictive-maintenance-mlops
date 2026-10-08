from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.model_registry import get_mlflow_client
from src.model_promotion import get_model_version_by_environment


def release_model_to_production(
    model_name: str,
    version: str,
) -> None:
    """
    Release an approved staging model to production.

    A model must:
    1. Exist in the registry.
    2. Have validation_status='approved'.
    3. Currently be assigned to the staging alias.
    """

    client: MlflowClient = get_mlflow_client()

    # Retrieve the requested model version.
    try:
        model_version = client.get_model_version(
            name=model_name,
            version=version,
        )
    except MlflowException as exc:
        raise ValueError(
            f"Model '{model_name}' version '{version}' "
            "was not found."
        ) from exc

    # Verify model approval.
    validation_status = model_version.tags.get(
        "validation_status"
    )

    if validation_status != "approved":
        raise ValueError(
            f"Model version {version} cannot be released "
            "to production because it is not approved."
        )

    # Verify that the same version is currently in staging.
    try:
        staging_version = get_model_version_by_environment(
            model_name=model_name,
            environment="staging",
        )
    except ValueError as exc:
        raise ValueError(
            "Production release requires a model to be "
            "assigned to staging first."
        ) from exc

    if str(staging_version.version) != str(version):
        raise ValueError(
            f"Production release rejected. "
            f"Requested version {version}, but staging "
            f"currently contains version "
            f"{staging_version.version}."
        )

    # Assign the production alias.
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=version,
    )

    # Record production release metadata.
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="production_release_status",
        value="approved",
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="production_release_source",
        value="staging",
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="deployment_environment",
        value="production",
    )

    print(
        f"Model '{model_name}' version {version} "
        "released to production."
    )


def get_production_model(
    model_name: str,
):
    """Return the model version currently assigned to production."""

    return get_model_version_by_environment(
        model_name=model_name,
        environment="production",
    )


def print_release_status(
    model_name: str,
) -> None:
    """Display the current production release status."""

    try:
        production_model = get_production_model(
            model_name=model_name,
        )

        print(
            f"Production model version: "
            f"{production_model.version}"
        )

        print(
            f"Production release status: "
            f"{production_model.tags.get('production_release_status')}"
        )

        print(
            f"Production release source: "
            f"{production_model.tags.get('production_release_source')}"
        )

    except ValueError:
        print("Production model: Not released")