from typing import Any
from unittest.mock import Mock, mock_open

import pytest
from model_engine_server.common.dtos.docker_repository import BuildImageResponse
from model_engine_server.common.dtos.endpoint_builder import (
    BuildEndpointRequest,
    BuildEndpointResponse,
    BuildEndpointStatus,
)
from model_engine_server.core.fake_notification_gateway import FakeNotificationGateway
from model_engine_server.core.notification_gateway import NotificationApp
from model_engine_server.domain.entities.model_bundle_entity import (
    ArtifactLike,
    RunnableImageFlavor,
)
from model_engine_server.domain.exceptions import (
    DockerBuildFailedException,
    EndpointResourceInfraException,
)
from model_engine_server.infra.gateways.fake_monitoring_metrics_gateway import (
    FakeMonitoringMetricsGateway,
)
from model_engine_server.infra.repositories import ModelEndpointCacheRepository
from model_engine_server.infra.services import (
    LiveEndpointBuilderService,
    live_endpoint_builder_service,
)


@pytest.fixture
def endpoint_builder_service_empty_docker_built(
    fake_docker_repository_image_always_exists,
    fake_resource_gateway,
    fake_monitoring_metrics_gateway,
    fake_model_endpoint_record_repository,
    fake_model_endpoint_cache_repository,
    fake_filesystem_gateway,
    fake_notification_gateway,
    fake_feature_flag_repository,
) -> LiveEndpointBuilderService:
    return LiveEndpointBuilderService(
        docker_repository=fake_docker_repository_image_always_exists,
        resource_gateway=fake_resource_gateway,
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        model_endpoint_record_repository=fake_model_endpoint_record_repository,
        model_endpoint_cache_repository=fake_model_endpoint_cache_repository,
        filesystem_gateway=fake_filesystem_gateway,
        notification_gateway=fake_notification_gateway,
        feature_flag_repo=fake_feature_flag_repository,
    )


@pytest.fixture
def endpoint_builder_service_empty_docker_not_built(
    fake_docker_repository_image_never_exists,
    fake_resource_gateway,
    fake_monitoring_metrics_gateway,
    fake_model_endpoint_record_repository,
    fake_model_endpoint_cache_repository,
    fake_filesystem_gateway,
    fake_notification_gateway,
    fake_feature_flag_repository,
) -> LiveEndpointBuilderService:
    return LiveEndpointBuilderService(
        docker_repository=fake_docker_repository_image_never_exists,
        resource_gateway=fake_resource_gateway,
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        model_endpoint_record_repository=fake_model_endpoint_record_repository,
        model_endpoint_cache_repository=fake_model_endpoint_cache_repository,
        filesystem_gateway=fake_filesystem_gateway,
        notification_gateway=fake_notification_gateway,
        feature_flag_repo=fake_feature_flag_repository,
    )


@pytest.fixture
def endpoint_builder_service_empty_docker_builds_dont_work(
    fake_docker_repository_image_never_exists_and_builds_dont_work,
    fake_resource_gateway,
    fake_monitoring_metrics_gateway,
    fake_model_endpoint_record_repository,
    fake_model_endpoint_cache_repository,
    fake_filesystem_gateway,
    fake_notification_gateway,
    fake_feature_flag_repository,
) -> LiveEndpointBuilderService:
    return LiveEndpointBuilderService(
        docker_repository=fake_docker_repository_image_never_exists_and_builds_dont_work,
        resource_gateway=fake_resource_gateway,
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        model_endpoint_record_repository=fake_model_endpoint_record_repository,
        model_endpoint_cache_repository=fake_model_endpoint_cache_repository,
        filesystem_gateway=fake_filesystem_gateway,
        notification_gateway=fake_notification_gateway,
        feature_flag_repo=fake_feature_flag_repository,
    )


@pytest.fixture(autouse=True)
def set_env_vars():
    live_endpoint_builder_service.ECR_AWS_PROFILE = "default"
    live_endpoint_builder_service.GIT_TAG = "test_tag"
    live_endpoint_builder_service.ENV = "test_env"
    live_endpoint_builder_service.open = mock_open()
    live_endpoint_builder_service.os.mkdir = Mock()


@pytest.mark.asyncio
async def test_build_endpoint(
    build_endpoint_request_sync_pytorch: BuildEndpointRequest,
    build_endpoint_request_async_custom: BuildEndpointRequest,
    build_endpoint_request_async_tensorflow: BuildEndpointRequest,
    build_endpoint_request_async_runnable_image: BuildEndpointRequest,
    build_endpoint_request_sync_runnable_image: BuildEndpointRequest,
    build_endpoint_request_streaming_runnable_image: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_built: LiveEndpointBuilderService,
    endpoint_builder_service_empty_docker_not_built: LiveEndpointBuilderService,
    fake_model_endpoint_cache_repository: ModelEndpointCacheRepository,
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    for service in [
        endpoint_builder_service_empty_docker_not_built,
        endpoint_builder_service_empty_docker_built,
    ]:
        repo: Any = service.model_endpoint_record_repository
        for request in [
            build_endpoint_request_async_tensorflow,
            build_endpoint_request_async_custom,
            build_endpoint_request_sync_pytorch,
            build_endpoint_request_async_runnable_image,
            build_endpoint_request_sync_runnable_image,
            build_endpoint_request_streaming_runnable_image,
        ]:
            fake_monitoring_metrics_gateway.reset()
            repo.add_model_endpoint_record(request.model_endpoint_record)
            # Pass in a deep copy of request since LiveEndpointBuilderService.convert_artifact_like_bundle_to_runnable_image mutate the request
            response = await service.build_endpoint(request.copy(deep=True))
            assert response == BuildEndpointResponse(status=BuildEndpointStatus.OK)
            assert fake_model_endpoint_cache_repository.read_endpoint_info(
                endpoint_id=request.model_endpoint_record.id,
                deployment_name=request.deployment_name,
            )
            assert fake_monitoring_metrics_gateway.attempted_build == 1
            assert fake_monitoring_metrics_gateway.docker_failed_build == 0
            assert fake_monitoring_metrics_gateway.successful_build == 1
            assert fake_monitoring_metrics_gateway.build_time_seconds > 0
            if isinstance(request.model_endpoint_record.current_model_bundle.flavor, ArtifactLike):
                if service == endpoint_builder_service_empty_docker_built:
                    assert sum(fake_monitoring_metrics_gateway.image_build_cache_hit.values()) > 0
                    assert sum(fake_monitoring_metrics_gateway.image_build_cache_miss.values()) == 0
                else:
                    assert sum(fake_monitoring_metrics_gateway.image_build_cache_hit.values()) == 0
                    assert sum(fake_monitoring_metrics_gateway.image_build_cache_miss.values()) > 0


@pytest.mark.asyncio
async def test_build_endpoint_update_failed_raises_resource_manager_exception(
    build_endpoint_request_sync_pytorch: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_built: LiveEndpointBuilderService,
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    repo: Any = endpoint_builder_service_empty_docker_built.model_endpoint_record_repository
    repo.add_model_endpoint_record(build_endpoint_request_sync_pytorch.model_endpoint_record)
    endpoint_builder_service_empty_docker_built.resource_gateway.__setattr__(
        "create_or_update_resources", Mock(side_effect=EndpointResourceInfraException)
    )
    with pytest.raises(EndpointResourceInfraException):
        await endpoint_builder_service_empty_docker_built.build_endpoint(
            build_endpoint_request_sync_pytorch
        )
        assert fake_monitoring_metrics_gateway.attempted_build == 1
        assert fake_monitoring_metrics_gateway.docker_failed_build == 0
        assert fake_monitoring_metrics_gateway.successful_build == 0


@pytest.mark.asyncio
async def test_build_endpoint_tensorflow_with_nonzero_gpu_raises_not_implemented(
    build_endpoint_request_async_tensorflow: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_not_built: LiveEndpointBuilderService,
):
    repo: Any = endpoint_builder_service_empty_docker_not_built.model_endpoint_record_repository
    repo.add_model_endpoint_record(build_endpoint_request_async_tensorflow.model_endpoint_record)
    build_endpoint_request_async_tensorflow.gpus = 1
    with pytest.raises(NotImplementedError):
        await endpoint_builder_service_empty_docker_not_built.build_endpoint(
            build_endpoint_request_async_tensorflow
        )


@pytest.mark.asyncio
async def test_build_endpoint_tensorflow_with_invalid_aws_role_raises_value_error(
    build_endpoint_request_async_tensorflow: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_not_built: LiveEndpointBuilderService,
):
    repo: Any = endpoint_builder_service_empty_docker_not_built.model_endpoint_record_repository
    repo.add_model_endpoint_record(build_endpoint_request_async_tensorflow.model_endpoint_record)
    build_endpoint_request_async_tensorflow.aws_role = "invalid_aws_role"
    with pytest.raises(ValueError):
        await endpoint_builder_service_empty_docker_not_built.build_endpoint(
            build_endpoint_request_async_tensorflow
        )


@pytest.mark.asyncio
async def test_build_endpoint_build_result_failed_yields_docker_build_failed_exception(
    build_endpoint_request_sync_pytorch: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_not_built: LiveEndpointBuilderService,
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
    fake_notification_gateway: FakeNotificationGateway,
):
    repo: Any = endpoint_builder_service_empty_docker_not_built.model_endpoint_record_repository
    repo.add_model_endpoint_record(build_endpoint_request_sync_pytorch.model_endpoint_record)
    endpoint_builder_service_empty_docker_not_built.docker_repository.__setattr__(
        "build_image",
        Mock(return_value=BuildImageResponse(status=False, logs="", job_name="")),
    )
    with pytest.raises(DockerBuildFailedException):
        await endpoint_builder_service_empty_docker_not_built.build_endpoint(
            build_endpoint_request_sync_pytorch
        )
    assert fake_monitoring_metrics_gateway.attempted_build == 1
    assert fake_monitoring_metrics_gateway.successful_build == 0
    assert fake_monitoring_metrics_gateway.docker_failed_build == 1
    assert len(fake_notification_gateway.notifications_sent[NotificationApp.SLACK]) == 1
    assert len(fake_notification_gateway.notifications_sent[NotificationApp.EMAIL]) == 1


@pytest.mark.asyncio
async def test_build_endpoint_build_result_throws_error_yields_docker_build_failed_exception(
    build_endpoint_request_sync_pytorch: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_builds_dont_work: LiveEndpointBuilderService,
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
    fake_notification_gateway: FakeNotificationGateway,
):
    repo: Any = (
        endpoint_builder_service_empty_docker_builds_dont_work.model_endpoint_record_repository
    )
    repo.add_model_endpoint_record(build_endpoint_request_sync_pytorch.model_endpoint_record)
    with pytest.raises(DockerBuildFailedException):
        await endpoint_builder_service_empty_docker_builds_dont_work.build_endpoint(
            build_endpoint_request_sync_pytorch
        )
    record = await repo.get_model_endpoint_record(
        build_endpoint_request_sync_pytorch.model_endpoint_record.id
    )
    assert record.status == "UPDATE_FAILED"
    assert fake_monitoring_metrics_gateway.attempted_build == 1
    assert fake_monitoring_metrics_gateway.successful_build == 0
    assert fake_monitoring_metrics_gateway.docker_failed_build == 1
    assert len(fake_notification_gateway.notifications_sent[NotificationApp.SLACK]) == 1
    assert len(fake_notification_gateway.notifications_sent[NotificationApp.EMAIL]) == 1


def test_convert_artifact_like_bundle_to_runnable_image(
    build_endpoint_request_sync_custom: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_built: LiveEndpointBuilderService,
):
    endpoint_builder_service_empty_docker_built.convert_artifact_like_bundle_to_runnable_image(
        build_endpoint_request_sync_custom, "test_repo", "test_tag"
    )

    new_bundle = build_endpoint_request_sync_custom.model_endpoint_record.current_model_bundle

    assert isinstance(new_bundle.flavor, RunnableImageFlavor)
    assert new_bundle.flavor.repository == "test_repo"
    assert new_bundle.flavor.tag == "test_tag"
    for env_key in {
        "OMP_NUM_THREADS",
        "BASE_PATH",
        "BUNDLE_URL",
        "AWS_PROFILE",
        "RESULTS_S3_BUCKET",
        "CHILD_FN_INFO",
        "PREWARM",
        "PORT",
        "ML_INFRA_SERVICES_CONFIG_PATH",
        "LOAD_PREDICT_FN_MODULE_PATH",
        "LOAD_MODEL_FN_MODULE_PATH",
    }:
        assert env_key in new_bundle.flavor.env

    assert len(new_bundle.flavor.command) > 0
