from typing import Any
from unittest.mock import Mock, mock_open

import pytest
from model_engine_server.common.dtos.endpoint_builder import (
    BuildEndpointRequest,
    BuildEndpointResponse,
    BuildEndpointStatus,
)
from model_engine_server.core.config import infra_config
from model_engine_server.domain.entities.model_bundle_entity import (
    ArtifactLike,
    RunnableImageFlavor,
)
from model_engine_server.domain.entities.model_endpoint_entity import ModelEndpointStatus
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


@pytest.fixture(autouse=True)
def set_env_vars():
    live_endpoint_builder_service.ECR_AWS_PROFILE = "default"
    live_endpoint_builder_service.GIT_TAG = "test_tag"
    live_endpoint_builder_service.ENV = "test_env"
    live_endpoint_builder_service.WORKSPACE_PATH = ".."
    live_endpoint_builder_service.BUILD_CONTEXT_TEMP_ROOT = "../model-engine/.build-context"
    live_endpoint_builder_service.open = mock_open()
    live_endpoint_builder_service.os.makedirs = Mock()
    live_endpoint_builder_service.open_wrapper = mock_open()
    live_endpoint_builder_service.tempfile.mkdtemp = Mock(
        return_value="../model-engine/.build-context/tmpdir"
    )
    live_endpoint_builder_service.tempfile.mkstemp = Mock(return_value=["", ""])


@pytest.mark.asyncio
async def test_build_endpoint(
    build_endpoint_request_sync_pytorch: BuildEndpointRequest,
    build_endpoint_request_async_custom: BuildEndpointRequest,
    build_endpoint_request_async_tensorflow: BuildEndpointRequest,
    build_endpoint_request_async_runnable_image: BuildEndpointRequest,
    build_endpoint_request_sync_runnable_image: BuildEndpointRequest,
    build_endpoint_request_streaming_runnable_image: BuildEndpointRequest,
    build_endpoint_request_async_zipartifact_highpri: BuildEndpointRequest,
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
            build_endpoint_request_async_zipartifact_highpri,
        ]:
            fake_monitoring_metrics_gateway.reset()
            repo.add_model_endpoint_record(request.model_endpoint_record)
            is_artifact = isinstance(
                request.model_endpoint_record.current_model_bundle.flavor, ArtifactLike
            )
            if is_artifact and service == endpoint_builder_service_empty_docker_not_built:
                with pytest.raises(DockerBuildFailedException):
                    await service.build_endpoint(request.copy(deep=True))
                continue
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
            if is_artifact:
                assert sum(fake_monitoring_metrics_gateway.image_build_cache_hit.values()) > 0
                assert sum(fake_monitoring_metrics_gateway.image_build_cache_miss.values()) == 0


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
async def test_build_endpoint_failure_persists_generic_status_reason(
    build_endpoint_request_sync_pytorch: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_built: LiveEndpointBuilderService,
):
    # A failed build records a status_reason; a non-DomainException (raw infra error) maps to
    # the generic message so internals aren't leaked.
    repo: Any = endpoint_builder_service_empty_docker_built.model_endpoint_record_repository
    record = build_endpoint_request_sync_pytorch.model_endpoint_record
    repo.add_model_endpoint_record(record)
    endpoint_builder_service_empty_docker_built.resource_gateway.__setattr__(
        "create_or_update_resources", Mock(side_effect=RuntimeError("raw k8s detail"))
    )

    with pytest.raises(RuntimeError):
        await endpoint_builder_service_empty_docker_built.build_endpoint(
            build_endpoint_request_sync_pytorch
        )

    persisted = await repo.get_model_endpoint_record(model_endpoint_id=record.id)
    assert persisted.status == ModelEndpointStatus.UPDATE_FAILED
    assert persisted.status_reason == live_endpoint_builder_service.GENERIC_STATUS_REASON


@pytest.mark.asyncio
async def test_build_endpoint_recovery_clears_prior_status_reason(
    build_endpoint_request_sync_pytorch: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_built: LiveEndpointBuilderService,
):
    # A successful build clears any status_reason left over from a prior failed attempt.
    repo: Any = endpoint_builder_service_empty_docker_built.model_endpoint_record_repository
    record = build_endpoint_request_sync_pytorch.model_endpoint_record
    record.status_reason = "Stale reason from a previous failed build."
    repo.add_model_endpoint_record(record)

    response = await endpoint_builder_service_empty_docker_built.build_endpoint(
        build_endpoint_request_sync_pytorch.copy(deep=True)
    )

    assert response == BuildEndpointResponse(status=BuildEndpointStatus.OK)
    persisted = await repo.get_model_endpoint_record(model_endpoint_id=record.id)
    assert persisted.status == ModelEndpointStatus.READY
    assert persisted.status_reason is None


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
async def test_build_endpoint_with_control_plane_aws_role_raises_value_error(
    build_endpoint_request_sync_runnable_image: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_built: LiveEndpointBuilderService,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(infra_config(), "profile_ml_worker", "control-plane-role")
    monkeypatch.setattr(infra_config(), "profile_ml_inference_worker", "inference-role")
    repo: Any = endpoint_builder_service_empty_docker_built.model_endpoint_record_repository
    repo.add_model_endpoint_record(build_endpoint_request_sync_runnable_image.model_endpoint_record)
    build_endpoint_request_sync_runnable_image.aws_role = "control-plane-role"
    with pytest.raises(ValueError):
        await endpoint_builder_service_empty_docker_built.build_endpoint(
            build_endpoint_request_sync_runnable_image
        )
    persisted = await repo.get_model_endpoint_record(
        model_endpoint_id=build_endpoint_request_sync_runnable_image.model_endpoint_record.id
    )
    assert persisted.status == ModelEndpointStatus.UPDATE_FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize("forged_field", ["owner", "created_by"])
async def test_build_endpoint_rejects_request_not_matching_stored_record(
    build_endpoint_request_sync_runnable_image: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_built: LiveEndpointBuilderService,
    forged_field,
):
    repo: Any = endpoint_builder_service_empty_docker_built.model_endpoint_record_repository
    request_record = build_endpoint_request_sync_runnable_image.model_endpoint_record
    repo.add_model_endpoint_record(request_record.copy(deep=True))
    setattr(request_record, forged_field, "forged")
    create_or_update_resources = Mock()
    endpoint_builder_service_empty_docker_built.resource_gateway.__setattr__(
        "create_or_update_resources", create_or_update_resources
    )

    with pytest.raises(ValueError):
        await endpoint_builder_service_empty_docker_built.build_endpoint(
            build_endpoint_request_sync_runnable_image
        )

    create_or_update_resources.assert_not_called()


@pytest.mark.asyncio
async def test_build_endpoint_checks_only_the_deployed_artifact_image(
    build_endpoint_request_sync_pytorch: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_built: LiveEndpointBuilderService,
):
    repo: Any = endpoint_builder_service_empty_docker_built.model_endpoint_record_repository
    repo.add_model_endpoint_record(build_endpoint_request_sync_pytorch.model_endpoint_record)
    image_exists = Mock(return_value=True)
    endpoint_builder_service_empty_docker_built.docker_repository.__setattr__(
        "image_exists", image_exists
    )

    await endpoint_builder_service_empty_docker_built.build_endpoint(
        build_endpoint_request_sync_pytorch
    )

    # Only the user image is checked: a deleted base image tag must not block the deploy.
    image_exists.assert_called_once()


@pytest.mark.asyncio
async def test_build_endpoint_skips_request_for_replaced_bundle(
    build_endpoint_request_sync_runnable_image: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_built: LiveEndpointBuilderService,
):
    repo: Any = endpoint_builder_service_empty_docker_built.model_endpoint_record_repository
    request_record = build_endpoint_request_sync_runnable_image.model_endpoint_record
    stored_record = request_record.copy(deep=True)
    stored_record.status = ModelEndpointStatus.READY
    repo.add_model_endpoint_record(stored_record)
    request_record.current_model_bundle = request_record.current_model_bundle.copy(
        update={"id": "replaced-bundle-id"}
    )
    create_or_update_resources = Mock()
    endpoint_builder_service_empty_docker_built.resource_gateway.__setattr__(
        "create_or_update_resources", create_or_update_resources
    )

    response = await endpoint_builder_service_empty_docker_built.build_endpoint(
        build_endpoint_request_sync_runnable_image
    )

    assert response.status == BuildEndpointStatus.OK
    create_or_update_resources.assert_not_called()
    persisted = await repo.get_model_endpoint_record(model_endpoint_id=request_record.id)
    assert persisted.status == ModelEndpointStatus.READY


@pytest.mark.asyncio
async def test_build_endpoint_rejects_tensorrt_llm_bundle(
    build_endpoint_request_sync_runnable_image: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_built: LiveEndpointBuilderService,
):
    repo: Any = endpoint_builder_service_empty_docker_built.model_endpoint_record_repository
    request_record = build_endpoint_request_sync_runnable_image.model_endpoint_record
    bundle = request_record.current_model_bundle
    request_record.current_model_bundle = bundle.copy(
        update={
            "flavor": bundle.flavor.copy(
                update={
                    "command": [
                        "/usr/bin/bash",
                        "-c",
                        "./s5cmd cp s3://bucket/ckpt/* ./ ; python3 launch_triton_server.py",
                    ]
                }
            )
        }
    )
    repo.add_model_endpoint_record(request_record.copy(deep=True))
    create_or_update_resources = Mock()
    endpoint_builder_service_empty_docker_built.resource_gateway.__setattr__(
        "create_or_update_resources", create_or_update_resources
    )

    with pytest.raises(ValueError):
        await endpoint_builder_service_empty_docker_built.build_endpoint(
            build_endpoint_request_sync_runnable_image
        )

    create_or_update_resources.assert_not_called()


@pytest.mark.asyncio
async def test_build_endpoint_deploys_stored_bundle(
    build_endpoint_request_sync_runnable_image: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_built: LiveEndpointBuilderService,
):
    repo: Any = endpoint_builder_service_empty_docker_built.model_endpoint_record_repository
    request_record = build_endpoint_request_sync_runnable_image.model_endpoint_record
    stored_record = request_record.copy(deep=True)
    repo.add_model_endpoint_record(stored_record)
    stored_bundle = stored_record.current_model_bundle
    request_record.current_model_bundle = stored_bundle.copy(
        update={"flavor": stored_bundle.flavor.copy(update={"tag": "forged-tag"})}
    )

    await endpoint_builder_service_empty_docker_built.build_endpoint(
        build_endpoint_request_sync_runnable_image
    )

    assert request_record.current_model_bundle == stored_bundle


@pytest.mark.asyncio
async def test_build_endpoint_missing_image_yields_docker_build_failed_exception(
    build_endpoint_request_sync_pytorch: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_not_built: LiveEndpointBuilderService,
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    service = endpoint_builder_service_empty_docker_not_built
    repo: Any = service.model_endpoint_record_repository
    record = build_endpoint_request_sync_pytorch.model_endpoint_record
    repo.add_model_endpoint_record(record)
    build_image = Mock()
    service.docker_repository.__setattr__("build_image", build_image)

    with pytest.raises(DockerBuildFailedException):
        await service.build_endpoint(build_endpoint_request_sync_pytorch)

    build_image.assert_not_called()
    persisted = await repo.get_model_endpoint_record(model_endpoint_id=record.id)
    assert persisted.status == ModelEndpointStatus.UPDATE_FAILED
    assert persisted.status_reason == (
        "Building images for cloudpickle and zip bundles is no longer supported. "
        "Use a runnable image bundle instead."
    )
    assert fake_monitoring_metrics_gateway.attempted_build == 1
    assert fake_monitoring_metrics_gateway.successful_build == 0
    assert fake_monitoring_metrics_gateway.docker_failed_build == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "running_bundle_url,running_requirements,reused",
    [
        pytest.param(None, None, True, id="same-bundle"),
        pytest.param("s3://other", None, False, id="other-bundle-url"),
        pytest.param(None, ["other==1.0"], False, id="other-requirements"),
    ],
)
async def test_build_endpoint_missing_image_reuses_running_image_for_same_bundle(
    build_endpoint_request_sync_pytorch: BuildEndpointRequest,
    endpoint_builder_service_empty_docker_not_built: LiveEndpointBuilderService,
    fake_resource_gateway,
    model_endpoint_1,
    running_bundle_url,
    running_requirements,
    reused: bool,
):
    service = endpoint_builder_service_empty_docker_not_built
    repo: Any = service.model_endpoint_record_repository
    request = build_endpoint_request_sync_pytorch
    record = request.model_endpoint_record
    repo.add_model_endpoint_record(record)
    # The image a builder at another GIT_TAG built for the running bundle.
    running_request = request.copy(deep=True)
    if running_requirements is not None:
        running_request.model_endpoint_record.current_model_bundle.requirements = (
            running_requirements
        )
    running_params = service._get_user_image_params("", running_request, Mock())
    running_image = service.docker_repository.get_image_url(
        running_params.image_tag.replace("-test_tag-", "-0ldgittag-"), running_params.repo
    )
    fake_resource_gateway.db[record.id] = model_endpoint_1.infra_state.copy(
        update=dict(
            image=running_image,
            bundle_url=running_bundle_url or record.current_model_bundle.flavor.location,
        )
    )

    if not reused:
        with pytest.raises(DockerBuildFailedException):
            await service.build_endpoint(request)
        return
    response = await service.build_endpoint(request)

    assert response == BuildEndpointResponse(status=BuildEndpointStatus.OK)
    assert fake_resource_gateway.db[record.id].image == running_image
    persisted = await repo.get_model_endpoint_record(model_endpoint_id=record.id)
    assert persisted.status == ModelEndpointStatus.READY


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
