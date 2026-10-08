from pipelines.production_inference_pipeline import (
    production_inference_pipeline,
)


if __name__ == "__main__":
    production_inference_pipeline(
        environment="production",
    )