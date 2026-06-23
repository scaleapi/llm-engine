from unittest.mock import MagicMock, patch

import pytest
from model_engine_server.api.dependencies import _get_external_interfaces
from model_engine_server.common.config import HostedModelInferenceServiceConfig
from model_engine_server.infra.gateways import (
    ABSFileStorageGateway,
    ABSFilesystemGateway,
    ABSLLMArtifactGateway,
    ASBInferenceAutoscalingMetricsGateway,
    GCSFileStorageGateway,
    GCSFilesystemGateway,
    GCSLLMArtifactGateway,
    RedisInferenceAutoscalingMetricsGateway,
    S3FilesystemGateway,
    S3LLMArtifactGateway,
)
from model_engine_server.infra.gateways.resources.asb_queue_endpoint_resource_delegate import (
    ASBQueueEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.resources.gcp_pubsub_queue_endpoint_resource_delegate import (
    GcpPubSubQueueEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.resources.onprem_queue_endpoint_resource_delegate import (
    OnPremQueueEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.resources.sqs_queue_endpoint_resource_delegate import (
    SQSQueueEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.s3_file_storage_gateway import S3FileStorageGateway
from model_engine_server.infra.repositories import (
    ABSFileLLMFineTuneEventsRepository,
    ABSFileLLMFineTuneRepository,
    ACRDockerRepository,
    ECRDockerRepository,
    GARDockerRepository,
    GCSFileLLMFineTuneEventsRepository,
    GCSFileLLMFineTuneRepository,
    OnPremDockerRepository,
    S3FileLLMFineTuneEventsRepository,
    S3FileLLMFineTuneRepository,
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


# Expected concrete backend class per cloud_provider. docker_repository is keyed on
# registry_type (not cloud_provider): aws/azure/gcp drive it via a realistic prefix, onprem
# via docker_registry_type since no prefix infers "onprem".
_PROVIDER_CASES = [
    pytest.param(
        "aws",
        "000000000000.dkr.ecr.us-east-1.amazonaws.com/my-repo",
        None,
        {
            "queue_delegate": SQSQueueEndpointResourceDelegate,
            "filesystem_gateway": S3FilesystemGateway,
            "llm_artifact_gateway": S3LLMArtifactGateway,
            "file_storage_gateway": S3FileStorageGateway,
            "docker_repository": ECRDockerRepository,
            "llm_fine_tune_repository": S3FileLLMFineTuneRepository,
            "llm_fine_tune_events_repository": S3FileLLMFineTuneEventsRepository,
            "inference_autoscaling_metrics_gateway": RedisInferenceAutoscalingMetricsGateway,
        },
        id="aws",
    ),
    pytest.param(
        "azure",
        "myregistry.azurecr.io/my-repo",
        None,
        {
            "queue_delegate": ASBQueueEndpointResourceDelegate,
            "filesystem_gateway": ABSFilesystemGateway,
            "llm_artifact_gateway": ABSLLMArtifactGateway,
            "file_storage_gateway": ABSFileStorageGateway,
            "docker_repository": ACRDockerRepository,
            "llm_fine_tune_repository": ABSFileLLMFineTuneRepository,
            "llm_fine_tune_events_repository": ABSFileLLMFineTuneEventsRepository,
            "inference_autoscaling_metrics_gateway": ASBInferenceAutoscalingMetricsGateway,
        },
        id="azure",
    ),
    pytest.param(
        "gcp",
        "us-docker.pkg.dev/my-project/my-repo",
        None,
        {
            "queue_delegate": GcpPubSubQueueEndpointResourceDelegate,
            "filesystem_gateway": GCSFilesystemGateway,
            "llm_artifact_gateway": GCSLLMArtifactGateway,
            "file_storage_gateway": GCSFileStorageGateway,
            "docker_repository": GARDockerRepository,
            "llm_fine_tune_repository": GCSFileLLMFineTuneRepository,
            "llm_fine_tune_events_repository": GCSFileLLMFineTuneEventsRepository,
            "inference_autoscaling_metrics_gateway": RedisInferenceAutoscalingMetricsGateway,
        },
        id="gcp",
    ),
    pytest.param(
        "onprem",
        "registry.internal/my-repo",
        "onprem",
        {
            "queue_delegate": OnPremQueueEndpointResourceDelegate,
            "filesystem_gateway": S3FilesystemGateway,
            "llm_artifact_gateway": S3LLMArtifactGateway,
            "file_storage_gateway": S3FileStorageGateway,
            "docker_repository": OnPremDockerRepository,
            "llm_fine_tune_repository": S3FileLLMFineTuneRepository,
            "llm_fine_tune_events_repository": S3FileLLMFineTuneEventsRepository,
            "inference_autoscaling_metrics_gateway": RedisInferenceAutoscalingMetricsGateway,
        },
        id="onprem",
    ),
]


@pytest.mark.parametrize(
    "cloud_provider, docker_repo_prefix, docker_registry_type, expected",
    _PROVIDER_CASES,
)
def test_cloud_provider_selects_expected_backends(
    cloud_provider, docker_repo_prefix, docker_registry_type, expected
):
    """Pin the concrete backend class selected for each cloud_provider."""
    with (
        patch("model_engine_server.api.dependencies.infra_config") as mock_config,
        patch("model_engine_server.api.dependencies.CIRCLECI", False),
        patch("model_engine_server.api.dependencies.CeleryTaskQueueGateway"),
        patch("model_engine_server.api.dependencies.get_tracing_gateway"),
        patch("model_engine_server.api.dependencies.aioredis"),
        patch("model_engine_server.api.dependencies.get_or_create_aioredis_pool"),
        patch("model_engine_server.api.dependencies.get_monitoring_metrics_gateway"),
    ):
        mock_config_instance = MagicMock()
        mock_config_instance.cloud_provider = cloud_provider
        mock_config_instance.docker_repo_prefix = docker_repo_prefix
        mock_config_instance.docker_registry_type = docker_registry_type
        mock_config_instance.celery_broker_type_redis = None
        mock_config_instance.gcp_project_id = "test-project"
        mock_config.return_value = mock_config_instance

        ei = _get_external_interfaces(read_only=False, session=MagicMock())

        assert isinstance(ei.resource_gateway.queue_delegate, expected["queue_delegate"])
        assert isinstance(ei.filesystem_gateway, expected["filesystem_gateway"])
        assert isinstance(ei.llm_artifact_gateway, expected["llm_artifact_gateway"])
        assert isinstance(ei.file_storage_gateway, expected["file_storage_gateway"])
        assert isinstance(ei.docker_repository, expected["docker_repository"])
        assert isinstance(
            ei.llm_fine_tuning_service.llm_fine_tune_repository,
            expected["llm_fine_tune_repository"],
        )
        assert isinstance(
            ei.llm_fine_tune_events_repository,
            expected["llm_fine_tune_events_repository"],
        )
        assert isinstance(
            ei.resource_gateway.inference_autoscaling_metrics_gateway,
            expected["inference_autoscaling_metrics_gateway"],
        )


def test_gcp_cache_redis_url_returns_gcp_url():
    """Test that cache_redis_url returns cache_redis_gcp_url when cloud_provider is gcp."""
    with patch("model_engine_server.common.config.infra_config") as mock_infra_config:
        mock_infra = MagicMock()
        mock_infra.cloud_provider = "gcp"
        mock_infra_config.return_value = mock_infra

        config = HostedModelInferenceServiceConfig(
            gateway_namespace="ns",
            endpoint_namespace="ns",
            billing_queue_arn="arn",
            sqs_profile="default",
            sqs_queue_policy_template="{}",
            sqs_queue_tag_template="{}",
            model_primitive_host="localhost",
            cloud_file_llm_fine_tune_repository="gs://bucket/ft",
            hf_user_fine_tuned_weights_prefix="gs://bucket/weights",
            istio_enabled=False,
            dd_trace_enabled=False,
            tgi_repository="repo/tgi",
            vllm_repository="repo/vllm",
            lightllm_repository="repo/lightllm",
            tensorrt_llm_repository="repo/tensorrt",
            batch_inference_vllm_repository="repo/batch",
            user_inference_base_repository="repo/base",
            user_inference_pytorch_repository="repo/pytorch",
            user_inference_tensorflow_repository="repo/tf",
            docker_image_layer_cache_repository="repo/cache",
            sensitive_log_mode=False,
            cache_redis_gcp_url="redis://10.0.0.1:6379/0",
        )
        assert config.cache_redis_url == "redis://10.0.0.1:6379/0"


def test_gcp_cache_redis_url_raises_when_not_set():
    """Test that cache_redis_url raises AssertionError for GCP when cache_redis_gcp_url is not set."""
    with patch("model_engine_server.common.config.infra_config") as mock_infra_config:
        mock_infra = MagicMock()
        mock_infra.cloud_provider = "gcp"
        mock_infra_config.return_value = mock_infra

        config = HostedModelInferenceServiceConfig(
            gateway_namespace="ns",
            endpoint_namespace="ns",
            billing_queue_arn="arn",
            sqs_profile="default",
            sqs_queue_policy_template="{}",
            sqs_queue_tag_template="{}",
            model_primitive_host="localhost",
            cloud_file_llm_fine_tune_repository="gs://bucket/ft",
            hf_user_fine_tuned_weights_prefix="gs://bucket/weights",
            istio_enabled=False,
            dd_trace_enabled=False,
            tgi_repository="repo/tgi",
            vllm_repository="repo/vllm",
            lightllm_repository="repo/lightllm",
            tensorrt_llm_repository="repo/tensorrt",
            batch_inference_vllm_repository="repo/batch",
            user_inference_base_repository="repo/base",
            user_inference_pytorch_repository="repo/pytorch",
            user_inference_tensorflow_repository="repo/tf",
            docker_image_layer_cache_repository="repo/cache",
            sensitive_log_mode=False,
        )
        with pytest.raises(AssertionError, match="cache_redis_gcp_url required for GCP"):
            _ = config.cache_redis_url
