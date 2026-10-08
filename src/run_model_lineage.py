from configs.pipeline_config import PipelineConfig

from src.model_lineage import print_model_lineage


def main() -> None:
    config = PipelineConfig()

    print_model_lineage(
        model_name=config.mlflow_model_name,
        version="3",
    )


if __name__ == "__main__":
    main()