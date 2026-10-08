from pipelines.model_registry_pipeline import (
    model_registry_pipeline,
)


if __name__ == "__main__":
    model_registry_pipeline(
        environment="production",
    )