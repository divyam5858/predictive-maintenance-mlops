from configs.pipeline_config import PipelineConfig

from src.model_promotion import print_environment_status
from src.model_release import (
    get_production_model,
    print_release_status,
    release_model_to_production,
)


def main() -> None:
    config = PipelineConfig()

    model_name = config.mlflow_model_name
    model_version = "3"

    print("Environment status before production release:")
    print_environment_status(
        model_name=model_name,
    )

    print("\nProduction release status:")
    print_release_status(
        model_name=model_name,
    )

    print(
        f"\nReleasing model version "
        f"{model_version} to production..."
    )

    release_model_to_production(
        model_name=model_name,
        version=model_version,
    )

    print("\nProduction model:")
    production_model = get_production_model(
        model_name=model_name,
    )

    print(
        f"Version: {production_model.version}"
    )

    print("\nEnvironment status after production release:")
    print_environment_status(
        model_name=model_name,
    )

    print("\nRelease metadata:")
    print(
        f"production_release_status: "
        f"{production_model.tags.get('production_release_status')}"
    )
    print(
        f"production_release_source: "
        f"{production_model.tags.get('production_release_source')}"
    )


if __name__ == "__main__":
    main()