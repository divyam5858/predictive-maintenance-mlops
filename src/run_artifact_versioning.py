from configs.pipeline_config import PipelineConfig

from src.artifact_versioning import (
    create_release_manifest,
    log_release_manifest,
)


def main() -> None:
    config = PipelineConfig()

    model_name = config.mlflow_model_name
    model_version = "3"
    environment = "production"

    print(
        "Creating production release artifact manifest..."
    )

    manifest = create_release_manifest(
        model_name=model_name,
        version=model_version,
        environment=environment,
    )

    print("\nRelease manifest:")
    print(manifest)

    run_id = log_release_manifest(
        manifest
    )

    print(
        f"\nRelease manifest logged successfully."
    )
    print(
        f"Artifact run ID: {run_id}"
    )
    print(
        "Artifact version: "
        f"{model_name}-v{model_version}"
    )


if __name__ == "__main__":
    main()