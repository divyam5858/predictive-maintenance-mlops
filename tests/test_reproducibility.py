from configs.pipeline_config import PipelineConfig


def test_pipeline_configuration_is_reproducible():
    """Verify that the pipeline uses deterministic configuration."""

    config_1 = PipelineConfig()
    config_2 = PipelineConfig()

    assert config_1.test_size == config_2.test_size
    assert config_1.random_state == config_2.random_state
    assert config_1.n_estimators == config_2.n_estimators
    assert config_1.class_weight == config_2.class_weight