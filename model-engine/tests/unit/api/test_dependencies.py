from unittest.mock import MagicMock, patch

from model_engine_server.api.dependencies import _get_external_interfaces
from model_engine_server.infra.gateways import (
    GCSFileStorageGateway,
    GCSFilesystemGateway,
    GCSLLMArtifactGateway,
)
from model_engine_server.infra.gateways.resources.redis_queue_endpoint_resource_delegate import (
    RedisQueueEndpointResourceDelegate,
)
from model_engine_server.infra.repositories import (
    GARDockerRepository,
    GCSFileLLMFineTuneEventsRepository,
)


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


def test_gcp_provider_selects_gcp_implementations():
    """Test that cloud_provider='gcp' wires the correct GCP implementations."""

    with (
        patch("model_engine_server.api.dependencies.infra_config") as mock_config,
        patch("model_engine_server.api.dependencies.CIRCLECI", False),
        patch("model_engine_server.api.dependencies.CeleryTaskQueueGateway"),
        patch("model_engine_server.api.dependencies.get_tracing_gateway"),
        patch("model_engine_server.api.dependencies.aioredis"),
        patch("model_engine_server.api.dependencies.get_or_create_aioredis_pool"),
        patch("model_engine_server.api.dependencies.ASBInferenceAutoscalingMetricsGateway"),
        patch("model_engine_server.api.dependencies.get_monitoring_metrics_gateway"),
    ):
        mock_config_instance = MagicMock()
        mock_config_instance.cloud_provider = "gcp"
        mock_config_instance.celery_broker_type_redis = None
        mock_config_instance.docker_repo_prefix = "us-docker.pkg.dev/my-project/my-repo"
        mock_config.return_value = mock_config_instance

        mock_session = MagicMock()
        external_interfaces = _get_external_interfaces(read_only=False, session=mock_session)

        assert isinstance(external_interfaces.filesystem_gateway, GCSFilesystemGateway)
        assert isinstance(external_interfaces.llm_artifact_gateway, GCSLLMArtifactGateway)
        assert isinstance(external_interfaces.file_storage_gateway, GCSFileStorageGateway)
        assert isinstance(external_interfaces.docker_repository, GARDockerRepository)
        assert isinstance(
            external_interfaces.llm_fine_tune_events_repository,
            GCSFileLLMFineTuneEventsRepository,
        )
        assert isinstance(
            external_interfaces.resource_gateway.queue_delegate,
            RedisQueueEndpointResourceDelegate,
        )
