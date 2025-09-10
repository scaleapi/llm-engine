from unittest.mock import MagicMock, patch

from model_engine_server.api.dependencies import _get_external_interfaces


def test_redis_task_queue_selection_when_celery_broker_type_redis_enabled():
    """Test that redis task queue gateways are selected when celery_broker_type_redis is True"""

    # Patch all the dependencies we need
    with (
        patch("model_engine_server.api.dependencies.infra_config") as mock_config,
        patch("model_engine_server.api.dependencies.CIRCLECI", False),
        patch("model_engine_server.api.dependencies.CeleryTaskQueueGateway") as mock_gateway_class,
        patch("model_engine_server.api.dependencies.get_tracing_gateway"),
        patch("model_engine_server.api.dependencies.aioredis"),
        patch("model_engine_server.api.dependencies.get_or_create_aioredis_pool"),
        patch("model_engine_server.api.dependencies.ASBInferenceAutoscalingMetricsGateway"),
        patch("model_engine_server.api.dependencies.get_monitoring_metrics_gateway"),
    ):

        # Mock the configuration
        mock_config_instance = MagicMock()
        mock_config_instance.celery_broker_type_redis = True
        mock_config_instance.cloud_provider = "aws"  # Not azure
        mock_config.return_value = mock_config_instance

        # Create different mock instances for each gateway
        redis_gateway = MagicMock()
        mock_gateway_class.return_value = redis_gateway

        # Create a mock session
        mock_session = MagicMock()

        # Call the actual function that contains the logic
        external_interfaces = _get_external_interfaces(read_only=False, session=mock_session)

        # Verify that the same redis gateway is used for both inference and infra
        assert external_interfaces.inference_task_queue_gateway == redis_gateway
        assert external_interfaces.infra_task_queue_gateway == redis_gateway


def test_default_task_queue_selection_when_celery_broker_type_redis_disabled():
    """Test that sqs task queue gateways are selected when celery_broker_type_redis is False"""

    with (
        patch("model_engine_server.api.dependencies.infra_config") as mock_config,
        patch("model_engine_server.api.dependencies.CIRCLECI", False),
        patch("model_engine_server.api.dependencies.CeleryTaskQueueGateway") as mock_gateway_class,
        patch("model_engine_server.api.dependencies.get_tracing_gateway"),
        patch("model_engine_server.api.dependencies.aioredis"),
        patch("model_engine_server.api.dependencies.get_or_create_aioredis_pool"),
        patch("model_engine_server.api.dependencies.ASBInferenceAutoscalingMetricsGateway"),
        patch("model_engine_server.api.dependencies.get_monitoring_metrics_gateway"),
    ):

        # Mock the configuration
        mock_config_instance = MagicMock()
        mock_config_instance.celery_broker_type_redis = False
        mock_config_instance.cloud_provider = "aws"  # Not azure, so should use SQS
        mock_config.return_value = mock_config_instance

        # Create different mock instances for each gateway
        sqs_gateway = MagicMock()
        mock_gateway_class.return_value = sqs_gateway

        # Create a mock session
        mock_session = MagicMock()

        # Call the actual function that contains the logic
        external_interfaces = _get_external_interfaces(read_only=False, session=mock_session)

        # Verify that the same sqs gateway is used for both inference and infra
        assert external_interfaces.inference_task_queue_gateway == sqs_gateway
        assert external_interfaces.infra_task_queue_gateway == sqs_gateway
