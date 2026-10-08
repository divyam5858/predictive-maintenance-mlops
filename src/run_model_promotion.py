from configs.pipeline_config import PipelineConfig
from src.model_promotion import (
    get_model_version_by_environment,
    print_environment_status,
    promote_model,
)


def main() -> None:
    config = PipelineConfig()

    print("Current environment assignments:")
    print_environment_status(
        model_name=config.mlflow_model_name,
    )

    print("\nPromoting accepted model to staging...")

    promote_model(
        model_name=config.mlflow_model_name,
        version="3",
        environment="staging",
    )

    staging_model = get_model_version_by_environment(
        model_name=config.mlflow_model_name,
        environment="staging",
    )

    print(
        f"\nStaging model version: "
        f"{staging_model.version}"
    )

    print("\nUpdated environment assignments:")
    print_environment_status(
        model_name=config.mlflow_model_name,
    )


if __name__ == "__main__":
    main()