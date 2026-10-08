from src.model_registry import get_mlflow_client


def get_model_lineage(
    model_name: str,
    version: str,
) -> dict[str, str | list[str]]:
    """Return lineage and lifecycle metadata for a model version."""

    client = get_mlflow_client()

    model_version = client.get_model_version(
        name=model_name,
        version=version,
    )

    tags = dict(model_version.tags)

    return {
        "model_name": model_version.name,
        "version": model_version.version,
        "run_id": model_version.run_id,
        "source": model_version.source,
        "aliases": list(model_version.aliases),
        "validation_status": tags.get(
            "validation_status",
            "unknown",
        ),
        "model_role": tags.get(
            "model_role",
            "unknown",
        ),
        "source_evaluation_run_id": tags.get(
            "source_evaluation_run_id",
            "unknown",
        ),
        "registration_run_id": tags.get(
            "registration_run_id",
            "unknown",
        ),
        "deployment_environment": tags.get(
            "deployment_environment",
            "unknown",
        ),
        "promotion_status": tags.get(
            "promotion_status",
            "unknown",
        ),
        "production_release_status": tags.get(
            "production_release_status",
            "unknown",
        ),
        "production_release_source": tags.get(
            "production_release_source",
            "unknown",
        ),
    }


def print_model_lineage(
    model_name: str,
    version: str,
) -> None:
    """Print a readable model lineage report."""

    lineage = get_model_lineage(
        model_name=model_name,
        version=version,
    )

    print("\nModel Lineage")
    print("=" * 70)

    for key, value in lineage.items():
        label = key.replace("_", " ").title()
        print(f"{label}: {value}")