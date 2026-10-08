from configs.pipeline_config import PipelineConfig

from src.registry_model_loader import (
    get_registered_model_uri,
    load_registered_model,
)


def main() -> None:
    config = PipelineConfig()

    print("Testing production model loading...")

    model_uri = get_registered_model_uri(
        environment="production",
    )

    print(f"Model URI: {model_uri}")

    model = load_registered_model(
        environment="production",
    )

    print(
        f"Pipeline steps: "
        f"{model.named_steps}"
    )

    print(
        "\nProduction model loaded successfully."
    )


if __name__ == "__main__":
    main()