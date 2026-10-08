import json
from datetime import datetime, timezone

import mlflow

from src.model_registry import get_mlflow_client
from src.model_lineage import get_model_lineage


def create_release_manifest(
    model_name: str,
    version: str,
    environment: str,
) -> dict:
    """
    Create a versioned release manifest for a registered model.
    """

    client = get_mlflow_client()

    model_version = client.get_model_version(
        name=model_name,
        version=version,
    )

    lineage = get_model_lineage(
        model_name=model_name,
        version=version,
    )

    manifest = {
        "release": {
            "model_name": model_name,
            "model_version": str(version),
            "environment": environment,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        },
        "model": {
            "source": model_version.source,
            "aliases": list(model_version.aliases),
            "validation_status": model_version.tags.get(
                "validation_status"
            ),
            "model_role": model_version.tags.get(
                "model_role"
            ),
        },
        "lineage": {
            "source_evaluation_run_id": lineage.get(
                "source_evaluation_run_id"
            ),
            "registration_run_id": lineage.get(
                "registration_run_id"
            ),
            "registry_run_id": lineage.get(
                "run_id"
            ),
        },
        "deployment": {
            "deployment_environment": model_version.tags.get(
                "deployment_environment"
            ),
            "promotion_status": model_version.tags.get(
                "promotion_status"
            ),
            "production_release_status": model_version.tags.get(
                "production_release_status"
            ),
            "production_release_source": model_version.tags.get(
                "production_release_source"
            ),
        },
    }

    return manifest


def log_release_manifest(
    manifest: dict,
) -> str:
    """
    Log a release manifest as an MLflow artifact.

    Returns the MLflow run ID containing the manifest.
    """

    model_name = manifest["release"]["model_name"]
    model_version = manifest["release"]["model_version"]
    environment = manifest["release"]["environment"]

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    with mlflow.start_run(
        run_name=(
            f"chapter9-artifact-release-"
            f"{model_name}-v{model_version}"
        )
    ) as run:

        mlflow.log_dict(
            manifest,
            "release_manifest.json",
        )

        mlflow.set_tag(
            "artifact_type",
            "model_release_manifest",
        )

        mlflow.set_tag(
            "model_name",
            model_name,
        )

        mlflow.set_tag(
            "model_version",
            model_version,
        )

        mlflow.set_tag(
            "deployment_environment",
            environment,
        )

        mlflow.set_tag(
            "artifact_version",
            f"{model_name}-v{model_version}",
        )

        return run.info.run_id