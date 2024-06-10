import json
from typing import Any, List, Tuple
from unittest import mock

import pytest
from model_engine_server.common.dtos.llms import (
    CompletionOutput,
    CompletionStreamV1Request,
    CompletionSyncV1Request,
    CreateBatchCompletionsRequest,
    CreateFineTuneRequest,
    CreateLLMModelEndpointV1Request,
    CreateLLMModelEndpointV1Response,
    ModelDownloadRequest,
    TokenOutput,
    UpdateLLMModelEndpointV1Request,
)
from model_engine_server.common.dtos.tasks import SyncEndpointPredictV1Response, TaskStatus
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.domain.entities import (
    LLMInferenceFramework,
    ModelEndpoint,
    ModelEndpointType,
)
from model_engine_server.domain.exceptions import (
    DockerImageNotFoundException,
    EndpointUnsupportedInferenceTypeException,
    InvalidRequestException,
    LLMFineTuningQuotaReached,
    ObjectHasInvalidValueException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
    UpstreamServiceError,
)
from model_engine_server.domain.use_cases.llm_fine_tuning_use_cases import (
    MAX_LLM_ENDPOINTS_PER_EXTERNAL_USER,
    CreateFineTuneV1UseCase,
    GetFineTuneEventsV1UseCase,
    is_model_name_suffix_valid,
)
from model_engine_server.domain.use_cases.llm_model_endpoint_use_cases import (
    CompletionStreamV1UseCase,
    CompletionSyncV1UseCase,
    CreateBatchCompletionsUseCase,
    CreateLLMModelBundleV1UseCase,
    CreateLLMModelEndpointV1UseCase,
    DeleteLLMEndpointByNameUseCase,
    GetLLMModelEndpointByNameV1UseCase,
    GpuType,
    ModelDownloadV1UseCase,
    UpdateLLMModelEndpointV1UseCase,
    _fill_hardware_info,
    _infer_hardware,
    validate_and_update_completion_params,
    validate_checkpoint_files,
)
from model_engine_server.domain.use_cases.model_bundle_use_cases import CreateModelBundleV2UseCase


def mocked__get_latest_tag():
    async def async_mock(*args, **kwargs):  # noqa
        return "fake_docker_repository_latest_image_tag"

    return mock.AsyncMock(side_effect=async_mock)


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.domain.use_cases.llm_model_endpoint_use_cases._get_latest_tag",
    mocked__get_latest_tag(),
)
async def test_create_model_endpoint_use_case_success(
    test_api_key: str,
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    fake_docker_repository_image_always_exists,
    fake_model_primitive_gateway,
    fake_llm_artifact_gateway,
    create_llm_model_endpoint_request_async: CreateLLMModelEndpointV1Request,
    create_llm_model_endpoint_request_sync: CreateLLMModelEndpointV1Request,
    create_llm_model_endpoint_request_streaming: CreateLLMModelEndpointV1Request,
    create_llm_model_endpoint_request_llama_2: CreateLLMModelEndpointV1Request,
    create_llm_model_endpoint_request_llama_3_70b: CreateLLMModelEndpointV1Request,
):
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    bundle_use_case = CreateModelBundleV2UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    llm_bundle_use_case = CreateLLMModelBundleV1UseCase(
        create_model_bundle_use_case=bundle_use_case,
        model_bundle_repository=fake_model_bundle_repository,
        llm_artifact_gateway=fake_llm_artifact_gateway,
        docker_repository=fake_docker_repository_image_always_exists,
    )
    use_case = CreateLLMModelEndpointV1UseCase(
        create_llm_model_bundle_use_case=llm_bundle_use_case,
        model_endpoint_service=fake_model_endpoint_service,
        docker_repository=fake_docker_repository_image_always_exists,
        llm_artifact_gateway=fake_llm_artifact_gateway,
    )

    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case.execute(user=user, request=create_llm_model_endpoint_request_async)
    assert response_1.endpoint_creation_task_id
    assert isinstance(response_1, CreateLLMModelEndpointV1Response)
    endpoint = (
        await fake_model_endpoint_service.list_model_endpoints(
            owner=None,
            name=create_llm_model_endpoint_request_async.name,
            order_by=None,
        )
    )[0]
    assert endpoint.record.endpoint_type == ModelEndpointType.ASYNC
    assert endpoint.record.metadata == {
        "_llm": {
            "model_name": create_llm_model_endpoint_request_async.model_name,
            "source": create_llm_model_endpoint_request_async.source,
            "inference_framework": create_llm_model_endpoint_request_async.inference_framework,
            "inference_framework_image_tag": "fake_docker_repository_latest_image_tag",
            "num_shards": create_llm_model_endpoint_request_async.num_shards,
            "quantize": None,
            "checkpoint_path": create_llm_model_endpoint_request_async.checkpoint_path,
        }
    }

    response_2 = await use_case.execute(user=user, request=create_llm_model_endpoint_request_sync)
    assert response_2.endpoint_creation_task_id
    assert isinstance(response_2, CreateLLMModelEndpointV1Response)
    endpoint = (
        await fake_model_endpoint_service.list_model_endpoints(
            owner=None,
            name=create_llm_model_endpoint_request_sync.name,
            order_by=None,
        )
    )[0]
    assert endpoint.record.endpoint_type == ModelEndpointType.SYNC
    assert endpoint.record.metadata == {
        "_llm": {
            "model_name": create_llm_model_endpoint_request_sync.model_name,
            "source": create_llm_model_endpoint_request_sync.source,
            "inference_framework": create_llm_model_endpoint_request_sync.inference_framework,
            "inference_framework_image_tag": create_llm_model_endpoint_request_sync.inference_framework_image_tag,
            "num_shards": create_llm_model_endpoint_request_sync.num_shards,
            "quantize": None,
            "checkpoint_path": create_llm_model_endpoint_request_sync.checkpoint_path,
        }
    }

    response_3 = await use_case.execute(
        user=user, request=create_llm_model_endpoint_request_streaming
    )
    assert response_3.endpoint_creation_task_id
    assert isinstance(response_3, CreateLLMModelEndpointV1Response)
    endpoint = (
        await fake_model_endpoint_service.list_model_endpoints(
            owner=None,
            name=create_llm_model_endpoint_request_streaming.name,
            order_by=None,
        )
    )[0]
    assert endpoint.record.endpoint_type == ModelEndpointType.STREAMING
    assert endpoint.record.metadata == {
        "_llm": {
            "model_name": create_llm_model_endpoint_request_streaming.model_name,
            "source": create_llm_model_endpoint_request_streaming.source,
            "inference_framework": create_llm_model_endpoint_request_streaming.inference_framework,
            "inference_framework_image_tag": create_llm_model_endpoint_request_streaming.inference_framework_image_tag,
            "num_shards": create_llm_model_endpoint_request_streaming.num_shards,
            "quantize": None,
            "checkpoint_path": create_llm_model_endpoint_request_sync.checkpoint_path,
        }
    }

    response_4 = await use_case.execute(
        user=user, request=create_llm_model_endpoint_request_llama_2
    )
    assert response_4.endpoint_creation_task_id
    assert isinstance(response_4, CreateLLMModelEndpointV1Response)
    bundle = await fake_model_bundle_repository.get_latest_model_bundle_by_name(
        owner=user.team_id, name=create_llm_model_endpoint_request_llama_2.name
    )
    assert "--max-total-tokens" in bundle.flavor.command[-1] and "4096" in bundle.flavor.command[-1]

    response_5 = await use_case.execute(
        user=user, request=create_llm_model_endpoint_request_llama_3_70b
    )
    assert response_5.endpoint_creation_task_id
    assert isinstance(response_5, CreateLLMModelEndpointV1Response)
    bundle = await fake_model_bundle_repository.get_latest_model_bundle_by_name(
        owner=user.team_id, name=create_llm_model_endpoint_request_llama_3_70b.name
    )
    assert " --gpu-memory-utilization 0.95" in bundle.flavor.command[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inference_framework, model_name, checkpoint_path, expected_error",
    [
        (LLMInferenceFramework.TEXT_GENERATION_INFERENCE, "mpt-7b", None, InvalidRequestException),
        (
            LLMInferenceFramework.TEXT_GENERATION_INFERENCE,
            "mpt-7b-instruct",
            "gibberish",
            ObjectHasInvalidValueException,
        ),
        (LLMInferenceFramework.LIGHTLLM, "mpt-7b", None, InvalidRequestException),
        (
            LLMInferenceFramework.LIGHTLLM,
            "mpt-7b-instruct",
            "gibberish",
            ObjectHasInvalidValueException,
        ),
        (LLMInferenceFramework.VLLM, "mpt-7b", None, InvalidRequestException),
        (
            LLMInferenceFramework.VLLM,
            "mpt-7b-instruct",
            "gibberish",
            ObjectHasInvalidValueException,
        ),
    ],
)
async def test_create_model_bundle_fails_if_no_checkpoint(
    test_api_key: str,
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    fake_docker_repository_image_always_exists,
    fake_model_primitive_gateway,
    fake_llm_artifact_gateway,
    create_llm_model_endpoint_text_generation_inference_request_streaming: CreateLLMModelEndpointV1Request,
    inference_framework,
    model_name,
    checkpoint_path,
    expected_error,
):
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    bundle_use_case = CreateModelBundleV2UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    use_case = CreateLLMModelBundleV1UseCase(
        create_model_bundle_use_case=bundle_use_case,
        model_bundle_repository=fake_model_bundle_repository,
        llm_artifact_gateway=fake_llm_artifact_gateway,
        docker_repository=fake_docker_repository_image_always_exists,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    request = create_llm_model_endpoint_text_generation_inference_request_streaming.copy()

    with pytest.raises(expected_error):
        await use_case.execute(
            user=user,
            endpoint_name=request.name,
            model_name=model_name,
            source=request.source,
            framework=inference_framework,
            framework_image_tag="0.0.0",
            endpoint_type=request.endpoint_type,
            num_shards=request.num_shards,
            quantize=request.quantize,
            checkpoint_path=checkpoint_path,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "valid, inference_framework, inference_framework_image_tag",
    [
        (False, LLMInferenceFramework.TEXT_GENERATION_INFERENCE, "0.9.2"),
        (True, LLMInferenceFramework.TEXT_GENERATION_INFERENCE, "0.9.3"),
        (False, LLMInferenceFramework.VLLM, "0.1.6"),
        (True, LLMInferenceFramework.VLLM, "0.1.3.6"),
    ],
)
async def test_create_model_bundle_inference_framework_image_tag_validation(
    test_api_key: str,
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    fake_docker_repository_image_always_exists,
    fake_docker_repository_image_never_exists,
    fake_model_primitive_gateway,
    fake_llm_artifact_gateway,
    create_llm_model_endpoint_text_generation_inference_request_streaming: CreateLLMModelEndpointV1Request,
    valid,
    inference_framework,
    inference_framework_image_tag,
):
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    bundle_use_case = CreateModelBundleV2UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    llm_bundle_use_case = CreateLLMModelBundleV1UseCase(
        create_model_bundle_use_case=bundle_use_case,
        model_bundle_repository=fake_model_bundle_repository,
        llm_artifact_gateway=fake_llm_artifact_gateway,
        docker_repository=fake_docker_repository_image_always_exists,
    )
    use_case = CreateLLMModelEndpointV1UseCase(
        create_llm_model_bundle_use_case=llm_bundle_use_case,
        model_endpoint_service=fake_model_endpoint_service,
        docker_repository=fake_docker_repository_image_always_exists,
        llm_artifact_gateway=fake_llm_artifact_gateway,
    )

    request = create_llm_model_endpoint_text_generation_inference_request_streaming.copy()
    request.inference_framework = inference_framework
    request.inference_framework_image_tag = inference_framework_image_tag
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    if valid:
        await use_case.execute(user=user, request=request)
    else:
        llm_bundle_use_case.docker_repository = fake_docker_repository_image_never_exists
        with pytest.raises(DockerImageNotFoundException):
            await use_case.execute(user=user, request=request)


@pytest.mark.asyncio
async def test_create_model_endpoint_text_generation_inference_use_case_success(
    test_api_key: str,
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    fake_docker_repository_image_always_exists,
    fake_model_primitive_gateway,
    fake_llm_artifact_gateway,
    create_llm_model_endpoint_text_generation_inference_request_async: CreateLLMModelEndpointV1Request,
    create_llm_model_endpoint_text_generation_inference_request_streaming: CreateLLMModelEndpointV1Request,
):
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    bundle_use_case = CreateModelBundleV2UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    llm_bundle_use_case = CreateLLMModelBundleV1UseCase(
        create_model_bundle_use_case=bundle_use_case,
        model_bundle_repository=fake_model_bundle_repository,
        llm_artifact_gateway=fake_llm_artifact_gateway,
        docker_repository=fake_docker_repository_image_always_exists,
    )
    use_case = CreateLLMModelEndpointV1UseCase(
        create_llm_model_bundle_use_case=llm_bundle_use_case,
        model_endpoint_service=fake_model_endpoint_service,
        docker_repository=fake_docker_repository_image_always_exists,
        llm_artifact_gateway=fake_llm_artifact_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case.execute(
        user=user,
        request=create_llm_model_endpoint_text_generation_inference_request_streaming,
    )
    assert response_1.endpoint_creation_task_id
    assert isinstance(response_1, CreateLLMModelEndpointV1Response)
    endpoint = (
        await fake_model_endpoint_service.list_model_endpoints(
            owner=None,
            name=create_llm_model_endpoint_text_generation_inference_request_streaming.name,
            order_by=None,
        )
    )[0]
    assert endpoint.record.endpoint_type == ModelEndpointType.STREAMING
    assert endpoint.record.metadata == {
        "_llm": {
            "model_name": create_llm_model_endpoint_text_generation_inference_request_streaming.model_name,
            "source": create_llm_model_endpoint_text_generation_inference_request_streaming.source,
            "inference_framework": create_llm_model_endpoint_text_generation_inference_request_streaming.inference_framework,
            "inference_framework_image_tag": create_llm_model_endpoint_text_generation_inference_request_streaming.inference_framework_image_tag,
            "num_shards": create_llm_model_endpoint_text_generation_inference_request_streaming.num_shards,
            "quantize": create_llm_model_endpoint_text_generation_inference_request_streaming.quantize,
            "checkpoint_path": create_llm_model_endpoint_text_generation_inference_request_streaming.checkpoint_path,
        }
    }

    with pytest.raises(ObjectHasInvalidValueException):
        await use_case.execute(
            user=user,
            request=create_llm_model_endpoint_text_generation_inference_request_async,
        )


def test_load_model_weights_sub_commands(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    fake_docker_repository_image_always_exists,
    fake_model_primitive_gateway,
    fake_llm_artifact_gateway,
):
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    bundle_use_case = CreateModelBundleV2UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    llm_bundle_use_case = CreateLLMModelBundleV1UseCase(
        create_model_bundle_use_case=bundle_use_case,
        model_bundle_repository=fake_model_bundle_repository,
        llm_artifact_gateway=fake_llm_artifact_gateway,
        docker_repository=fake_docker_repository_image_always_exists,
    )

    framework = LLMInferenceFramework.VLLM
    framework_image_tag = "0.2.7"
    checkpoint_path = "fake-checkpoint"
    final_weights_folder = "test_folder"

    subcommands = llm_bundle_use_case.load_model_weights_sub_commands(
        framework, framework_image_tag, checkpoint_path, final_weights_folder
    )

    expected_result = [
        "./s5cmd --numworkers 512 cp --concurrency 10 --include '*.model' --include '*.json' --include '*.safetensors' --exclude 'optimizer*' fake-checkpoint/* test_folder",
    ]
    assert expected_result == subcommands

    framework = LLMInferenceFramework.TEXT_GENERATION_INFERENCE
    framework_image_tag = "1.0.0"
    checkpoint_path = "fake-checkpoint"
    final_weights_folder = "test_folder"

    subcommands = llm_bundle_use_case.load_model_weights_sub_commands(
        framework, framework_image_tag, checkpoint_path, final_weights_folder
    )

    expected_result = [
        "s5cmd > /dev/null || conda install -c conda-forge -y s5cmd",
        "s5cmd --numworkers 512 cp --concurrency 10 --include '*.model' --include '*.json' --include '*.safetensors' --exclude 'optimizer*' fake-checkpoint/* test_folder",
    ]
    assert expected_result == subcommands


@pytest.mark.asyncio
async def test_create_model_endpoint_trt_llm_use_case_success(
    test_api_key: str,
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    fake_docker_repository_image_always_exists,
    fake_model_primitive_gateway,
    fake_llm_artifact_gateway,
    create_llm_model_endpoint_trt_llm_request_async: CreateLLMModelEndpointV1Request,
    create_llm_model_endpoint_trt_llm_request_streaming: CreateLLMModelEndpointV1Request,
):
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    bundle_use_case = CreateModelBundleV2UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    llm_bundle_use_case = CreateLLMModelBundleV1UseCase(
        create_model_bundle_use_case=bundle_use_case,
        model_bundle_repository=fake_model_bundle_repository,
        llm_artifact_gateway=fake_llm_artifact_gateway,
        docker_repository=fake_docker_repository_image_always_exists,
    )
    use_case = CreateLLMModelEndpointV1UseCase(
        create_llm_model_bundle_use_case=llm_bundle_use_case,
        model_endpoint_service=fake_model_endpoint_service,
        docker_repository=fake_docker_repository_image_always_exists,
        llm_artifact_gateway=fake_llm_artifact_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case.execute(
        user=user,
        request=create_llm_model_endpoint_trt_llm_request_streaming,
    )
    assert response_1.endpoint_creation_task_id
    assert isinstance(response_1, CreateLLMModelEndpointV1Response)
    endpoint = (
        await fake_model_endpoint_service.list_model_endpoints(
            owner=None,
            name=create_llm_model_endpoint_trt_llm_request_streaming.name,
            order_by=None,
        )
    )[0]
    assert endpoint.record.endpoint_type == ModelEndpointType.STREAMING
    assert endpoint.record.metadata == {
        "_llm": {
            "model_name": create_llm_model_endpoint_trt_llm_request_streaming.model_name,
            "source": create_llm_model_endpoint_trt_llm_request_streaming.source,
            "inference_framework": create_llm_model_endpoint_trt_llm_request_streaming.inference_framework,
            "inference_framework_image_tag": create_llm_model_endpoint_trt_llm_request_streaming.inference_framework_image_tag,
            "num_shards": create_llm_model_endpoint_trt_llm_request_streaming.num_shards,
            "quantize": create_llm_model_endpoint_trt_llm_request_streaming.quantize,
            "checkpoint_path": create_llm_model_endpoint_trt_llm_request_streaming.checkpoint_path,
        }
    }

    with pytest.raises(ObjectHasInvalidValueException):
        await use_case.execute(
            user=user,
            request=create_llm_model_endpoint_trt_llm_request_async,
        )


@pytest.mark.asyncio
async def test_create_llm_model_endpoint_use_case_raises_invalid_value_exception(
    test_api_key: str,
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    fake_docker_repository_image_always_exists,
    fake_model_primitive_gateway,
    fake_llm_artifact_gateway,
    create_llm_model_endpoint_request_invalid_model_name: CreateLLMModelEndpointV1Request,
):
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    bundle_use_case = CreateModelBundleV2UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    llm_bundle_use_case = CreateLLMModelBundleV1UseCase(
        create_model_bundle_use_case=bundle_use_case,
        model_bundle_repository=fake_model_bundle_repository,
        llm_artifact_gateway=fake_llm_artifact_gateway,
        docker_repository=fake_docker_repository_image_always_exists,
    )
    use_case = CreateLLMModelEndpointV1UseCase(
        create_llm_model_bundle_use_case=llm_bundle_use_case,
        model_endpoint_service=fake_model_endpoint_service,
        docker_repository=fake_docker_repository_image_always_exists,
        llm_artifact_gateway=fake_llm_artifact_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    with pytest.raises(ObjectHasInvalidValueException):
        await use_case.execute(
            user=user, request=create_llm_model_endpoint_request_invalid_model_name
        )


@pytest.mark.asyncio
async def test_create_llm_model_endpoint_use_case_quantization_exception(
    test_api_key: str,
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    fake_docker_repository_image_always_exists,
    fake_model_primitive_gateway,
    fake_llm_artifact_gateway,
    create_llm_model_endpoint_request_invalid_quantization: CreateLLMModelEndpointV1Request,
):
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    bundle_use_case = CreateModelBundleV2UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    llm_bundle_use_case = CreateLLMModelBundleV1UseCase(
        create_model_bundle_use_case=bundle_use_case,
        model_bundle_repository=fake_model_bundle_repository,
        llm_artifact_gateway=fake_llm_artifact_gateway,
        docker_repository=fake_docker_repository_image_always_exists,
    )
    use_case = CreateLLMModelEndpointV1UseCase(
        create_llm_model_bundle_use_case=llm_bundle_use_case,
        model_endpoint_service=fake_model_endpoint_service,
        docker_repository=fake_docker_repository_image_always_exists,
        llm_artifact_gateway=fake_llm_artifact_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    with pytest.raises(ObjectHasInvalidValueException):
        await use_case.execute(
            user=user, request=create_llm_model_endpoint_request_invalid_quantization
        )


@pytest.mark.asyncio
async def test_get_llm_model_endpoint_use_case_raises_not_found(
    test_api_key: str,
    fake_llm_model_endpoint_service,
    llm_model_endpoint_async: Tuple[ModelEndpoint, Any],
):
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_async[0])
    use_case = GetLLMModelEndpointByNameV1UseCase(
        llm_model_endpoint_service=fake_llm_model_endpoint_service
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    with pytest.raises(ObjectNotFoundException):
        await use_case.execute(user=user, model_endpoint_name="invalid_model_endpoint_name")


@pytest.mark.asyncio
async def test_get_llm_model_endpoint_use_case_raises_not_authorized(
    test_api_key: str,
    fake_llm_model_endpoint_service,
    llm_model_endpoint_async: Tuple[ModelEndpoint, Any],
):
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_async[0])
    use_case = GetLLMModelEndpointByNameV1UseCase(
        llm_model_endpoint_service=fake_llm_model_endpoint_service
    )
    llm_model_endpoint_async[0].record.public_inference = False
    user = User(user_id="non_exist", team_id="non_exist", is_privileged_user=False)
    with pytest.raises(ObjectNotAuthorizedException):
        await use_case.execute(
            user=user, model_endpoint_name=llm_model_endpoint_async[0].record.name
        )


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.domain.use_cases.llm_model_endpoint_use_cases._get_latest_tag",
    mocked__get_latest_tag(),
)
async def test_update_model_endpoint_use_case_success(
    test_api_key: str,
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    fake_docker_repository_image_always_exists,
    fake_model_primitive_gateway,
    fake_llm_artifact_gateway,
    fake_llm_model_endpoint_service,
    create_llm_model_endpoint_request_streaming: CreateLLMModelEndpointV1Request,
    update_llm_model_endpoint_request: UpdateLLMModelEndpointV1Request,
):
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    bundle_use_case = CreateModelBundleV2UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    llm_bundle_use_case = CreateLLMModelBundleV1UseCase(
        create_model_bundle_use_case=bundle_use_case,
        model_bundle_repository=fake_model_bundle_repository,
        llm_artifact_gateway=fake_llm_artifact_gateway,
        docker_repository=fake_docker_repository_image_always_exists,
    )
    create_use_case = CreateLLMModelEndpointV1UseCase(
        create_llm_model_bundle_use_case=llm_bundle_use_case,
        model_endpoint_service=fake_model_endpoint_service,
        docker_repository=fake_docker_repository_image_always_exists,
        llm_artifact_gateway=fake_llm_artifact_gateway,
    )
    update_use_case = UpdateLLMModelEndpointV1UseCase(
        create_llm_model_bundle_use_case=llm_bundle_use_case,
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
        docker_repository=fake_docker_repository_image_always_exists,
    )

    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)

    await create_use_case.execute(user=user, request=create_llm_model_endpoint_request_streaming)
    endpoint = (
        await fake_model_endpoint_service.list_model_endpoints(
            owner=None,
            name=create_llm_model_endpoint_request_streaming.name,
            order_by=None,
        )
    )[0]
    fake_llm_model_endpoint_service.add_model_endpoint(endpoint)

    update_response = await update_use_case.execute(
        user=user,
        model_endpoint_name=create_llm_model_endpoint_request_streaming.name,
        request=update_llm_model_endpoint_request,
    )
    assert update_response.endpoint_creation_task_id
    endpoint = (
        await fake_model_endpoint_service.list_model_endpoints(
            owner=None,
            name=create_llm_model_endpoint_request_streaming.name,
            order_by=None,
        )
    )[0]
    assert endpoint.record.endpoint_type == ModelEndpointType.STREAMING
    assert endpoint.record.metadata == {
        "_llm": {
            "model_name": create_llm_model_endpoint_request_streaming.model_name,
            "source": create_llm_model_endpoint_request_streaming.source,
            "inference_framework": create_llm_model_endpoint_request_streaming.inference_framework,
            "inference_framework_image_tag": "fake_docker_repository_latest_image_tag",
            "num_shards": create_llm_model_endpoint_request_streaming.num_shards,
            "quantize": None,
            "checkpoint_path": update_llm_model_endpoint_request.checkpoint_path,
        }
    }
    assert endpoint.infra_state.resource_state.memory == update_llm_model_endpoint_request.memory
    assert (
        endpoint.infra_state.deployment_state.min_workers
        == update_llm_model_endpoint_request.min_workers
    )
    assert (
        endpoint.infra_state.deployment_state.max_workers
        == update_llm_model_endpoint_request.max_workers
    )


def mocked_auto_tokenizer_from_pretrained(*args, **kwargs):  # noqa
    class mocked_encode:
        def encode(self, input: str) -> List[Any]:  # noqa
            return [1] * 7

    return mocked_encode()


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.infra.repositories.live_tokenizer_repository.AutoTokenizer.from_pretrained",
    mocked_auto_tokenizer_from_pretrained,
)
async def test_completion_sync_use_case_success(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    fake_tokenizer_repository,
    llm_model_endpoint_sync: Tuple[ModelEndpoint, Any],
    completion_sync_request: CompletionSyncV1Request,
):
    completion_sync_request.include_stop_str_in_output = True
    completion_sync_request.guided_json = {}
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_sync[0])
    fake_model_endpoint_service.sync_model_endpoint_inference_gateway.response = (
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={
                "result": json.dumps(
                    {
                        "text": "I am a newbie to the world of programming.",
                        "tokens": [
                            "I",
                            " am",
                            " a",
                            " new",
                            "bie",
                            " to",
                            " the",
                            " world",
                            " of",
                            " programming",
                            ".",
                        ],
                        "log_probs": [
                            {1: -2.3025850929940455},
                            {1: 0},
                            {1: 0},
                            {1: 0},
                            {1: 0},
                            {1: 0},
                            {1: 0},
                            {1: 0},
                            {1: 0},
                            {1: 0},
                            {1: 0},
                        ],
                        "count_prompt_tokens": 7,
                        "count_output_tokens": 11,
                    }
                )
            },
            traceback=None,
        )
    )
    use_case = CompletionSyncV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
        tokenizer_repository=fake_tokenizer_repository,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case.execute(
        user=user,
        model_endpoint_name=llm_model_endpoint_sync[0].record.name,
        request=completion_sync_request,
    )
    assert response_1.output == CompletionOutput(
        text="I am a newbie to the world of programming.",
        num_prompt_tokens=7,
        num_completion_tokens=11,
        tokens=[
            TokenOutput(token="I", log_prob=-2.3025850929940455),
            TokenOutput(token=" am", log_prob=0),
            TokenOutput(token=" a", log_prob=0),
            TokenOutput(token=" new", log_prob=0),
            TokenOutput(token="bie", log_prob=0),
            TokenOutput(token=" to", log_prob=0),
            TokenOutput(token=" the", log_prob=0),
            TokenOutput(token=" world", log_prob=0),
            TokenOutput(token=" of", log_prob=0),
            TokenOutput(token=" programming", log_prob=0),
            TokenOutput(token=".", log_prob=0),
        ],
    )


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.domain.use_cases.llm_model_endpoint_use_cases.count_tokens",
    return_value=5,
)
async def test_completion_sync_text_generation_inference_use_case_success(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    fake_tokenizer_repository,
    llm_model_endpoint_text_generation_inference: ModelEndpoint,
    completion_sync_request: CompletionSyncV1Request,
):
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_text_generation_inference)
    fake_model_endpoint_service.sync_model_endpoint_inference_gateway.response = SyncEndpointPredictV1Response(
        status=TaskStatus.SUCCESS,
        result={
            "result": """
  {
    "generated_text": " Deep Learning is a new type of machine learning",
    "details": {
      "finish_reason": "length",
      "generated_tokens": 9,
      "prefill": [
        {
          "id": 10560,
          "text": "What"
        },
        {
          "id": 632,
          "text": " is"
        },
        {
          "id": 89554,
          "text": " Deep"
        },
        {
          "id": 89950,
          "text": " Learning"
        },
        {
          "id": 34,
          "text": "?"
        }
      ],
      "tokens": [
        {
          "text": " Deep",
          "logprob": 0
        },
        {
          "text": " Learning",
          "logprob": -1
        },
        {
          "text": " is",
          "logprob": 0
        },
        {
          "text": " a",
          "logprob": 0
        },
        {
          "text": " new",
          "logprob": 0
        },
        {
          "text": " type",
          "logprob": 0
        },
        {
          "text": " of",
          "logprob": 0
        },
        {
          "text": " machine",
          "logprob": 0
        },
        {
          "text": " learning",
          "logprob": 0
        }
      ]
    }
  }
"""
        },
        traceback=None,
    )
    use_case = CompletionSyncV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
        tokenizer_repository=fake_tokenizer_repository,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case.execute(
        user=user,
        model_endpoint_name=llm_model_endpoint_text_generation_inference.record.name,
        request=completion_sync_request,
    )
    assert response_1.output == CompletionOutput(
        text=" Deep Learning is a new type of machine learning",
        num_prompt_tokens=5,
        num_completion_tokens=9,
        tokens=[
            TokenOutput(token=" Deep", log_prob=0.0),
            TokenOutput(token=" Learning", log_prob=-1.0),
            TokenOutput(token=" is", log_prob=0.0),
            TokenOutput(token=" a", log_prob=0.0),
            TokenOutput(token=" new", log_prob=0.0),
            TokenOutput(token=" type", log_prob=0.0),
            TokenOutput(token=" of", log_prob=0.0),
            TokenOutput(token=" machine", log_prob=0.0),
            TokenOutput(token=" learning", log_prob=0.0),
        ],
    )


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.domain.use_cases.llm_model_endpoint_use_cases.count_tokens",
    return_value=6,
)
async def test_completion_sync_trt_llm_use_case_success_23_10(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    fake_tokenizer_repository,
    llm_model_endpoint_trt_llm: ModelEndpoint,
    completion_sync_request: CompletionSyncV1Request,
):
    completion_sync_request.return_token_log_probs = False  # not yet supported
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_trt_llm)
    fake_model_endpoint_service.sync_model_endpoint_inference_gateway.response = SyncEndpointPredictV1Response(
        status=TaskStatus.SUCCESS,
        result={
            "result": '{"model_name": "ensemble", "model_version": "1", "sequence_end": false, "sequence_id": 0, "sequence_start": false, "text_output": "<s> What is machine learning? Machine learning is a branch", "token_ids": [1, 1724, 338, 4933, 6509, 29973, 6189, 6509, 338, 263, 5443]}'
        },
        traceback=None,
    )
    use_case = CompletionSyncV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
        tokenizer_repository=fake_tokenizer_repository,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case.execute(
        user=user,
        model_endpoint_name=llm_model_endpoint_trt_llm.record.name,
        request=completion_sync_request,
    )
    assert response_1.output == CompletionOutput(
        text=" Machine learning is a branch",
        num_prompt_tokens=6,
        num_completion_tokens=5,
    )


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.domain.use_cases.llm_model_endpoint_use_cases.count_tokens",
    return_value=6,
)
@pytest.mark.parametrize(
    "output_log_probs,output_tokens", [("[0.0,0.0,0.0,0.0,0.0]", 5), ("0.0", 1)]
)
async def test_completion_sync_trt_llm_use_case_success_24_01(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    fake_tokenizer_repository,
    llm_model_endpoint_trt_llm: ModelEndpoint,
    completion_sync_request: CompletionSyncV1Request,
    output_log_probs: str,
    output_tokens: int,
):
    completion_sync_request.return_token_log_probs = False  # not yet supported
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_trt_llm)
    fake_model_endpoint_service.sync_model_endpoint_inference_gateway.response = SyncEndpointPredictV1Response(
        status=TaskStatus.SUCCESS,
        result={
            "result": f'{{"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":{output_log_probs},"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" Machine learning is a branch"}}'
        },
        traceback=None,
    )
    use_case = CompletionSyncV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
        tokenizer_repository=fake_tokenizer_repository,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case.execute(
        user=user,
        model_endpoint_name=llm_model_endpoint_trt_llm.record.name,
        request=completion_sync_request,
    )
    assert response_1.output == CompletionOutput(
        text=" Machine learning is a branch",
        num_prompt_tokens=6,
        num_completion_tokens=output_tokens,
    )


@pytest.mark.asyncio
async def test_completion_sync_use_case_predict_failed(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    fake_tokenizer_repository,
    llm_model_endpoint_sync: Tuple[ModelEndpoint, Any],
    completion_sync_request: CompletionSyncV1Request,
):
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_sync[0])
    fake_model_endpoint_service.sync_model_endpoint_inference_gateway.response = (
        SyncEndpointPredictV1Response(
            status=TaskStatus.FAILURE,
            result=None,
            traceback="failed to predict",
        )
    )
    use_case = CompletionSyncV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
        tokenizer_repository=fake_tokenizer_repository,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    with pytest.raises(UpstreamServiceError):
        await use_case.execute(
            user=user,
            model_endpoint_name=llm_model_endpoint_sync[0].record.name,
            request=completion_sync_request,
        )


@pytest.mark.asyncio
async def test_completion_sync_use_case_predict_failed_lightllm(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    fake_tokenizer_repository,
    llm_model_endpoint_sync_lightllm: Tuple[ModelEndpoint, Any],
    completion_sync_request: CompletionSyncV1Request,
):
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_sync_lightllm[0])
    fake_model_endpoint_service.sync_model_endpoint_inference_gateway.response = (
        SyncEndpointPredictV1Response(
            status=TaskStatus.FAILURE,
            result=None,
            traceback="failed to predict",
        )
    )
    use_case = CompletionSyncV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
        tokenizer_repository=fake_tokenizer_repository,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    with pytest.raises(UpstreamServiceError):
        await use_case.execute(
            user=user,
            model_endpoint_name=llm_model_endpoint_sync_lightllm[0].record.name,
            request=completion_sync_request,
        )


@pytest.mark.asyncio
async def test_completion_sync_use_case_predict_failed_trt_llm(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    fake_tokenizer_repository,
    llm_model_endpoint_sync_trt_llm: Tuple[ModelEndpoint, Any],
    completion_sync_request: CompletionSyncV1Request,
):
    completion_sync_request.return_token_log_probs = False  # not yet supported
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_sync_trt_llm[0])
    fake_model_endpoint_service.sync_model_endpoint_inference_gateway.response = (
        SyncEndpointPredictV1Response(
            status=TaskStatus.FAILURE,
            result=None,
            traceback="failed to predict",
        )
    )
    use_case = CompletionSyncV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
        tokenizer_repository=fake_tokenizer_repository,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    with pytest.raises(UpstreamServiceError):
        await use_case.execute(
            user=user,
            model_endpoint_name=llm_model_endpoint_sync_trt_llm[0].record.name,
            request=completion_sync_request,
        )


@pytest.mark.asyncio
async def test_completion_sync_use_case_predict_failed_with_errors(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    fake_tokenizer_repository,
    llm_model_endpoint_sync_tgi: Tuple[ModelEndpoint, Any],
    completion_sync_request: CompletionSyncV1Request,
):
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_sync_tgi[0])
    fake_model_endpoint_service.sync_model_endpoint_inference_gateway.response = SyncEndpointPredictV1Response(
        status=TaskStatus.SUCCESS,
        result={
            "result": """
  {
    "error": "Request failed during generation: Server error: transport error",
    "error_type": "generation"
  }
"""
        },
        traceback="failed to predict",
    )
    use_case = CompletionSyncV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
        tokenizer_repository=fake_tokenizer_repository,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    with pytest.raises(UpstreamServiceError):
        await use_case.execute(
            user=user,
            model_endpoint_name=llm_model_endpoint_sync_tgi[0].record.name,
            request=completion_sync_request,
        )


@pytest.mark.asyncio
async def test_completion_sync_use_case_not_sync_endpoint_raises(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    fake_tokenizer_repository,
    llm_model_endpoint_async: Tuple[ModelEndpoint, Any],
    completion_sync_request: CompletionSyncV1Request,
):
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_async[0])
    use_case = CompletionSyncV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
        tokenizer_repository=fake_tokenizer_repository,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    with pytest.raises(EndpointUnsupportedInferenceTypeException):
        await use_case.execute(
            user=user,
            model_endpoint_name=llm_model_endpoint_async[0].record.name,
            request=completion_sync_request,
        )


@pytest.mark.asyncio
async def test_validate_and_update_completion_params():
    completion_sync_request = CompletionSyncV1Request(
        prompt="What is machine learning?",
        max_new_tokens=10,
        temperature=0.5,
        return_token_log_probs=True,
    )

    validate_and_update_completion_params(LLMInferenceFramework.VLLM, completion_sync_request)

    validate_and_update_completion_params(
        LLMInferenceFramework.TEXT_GENERATION_INFERENCE, completion_sync_request
    )

    completion_sync_request.include_stop_str_in_output = True
    with pytest.raises(ObjectHasInvalidValueException):
        validate_and_update_completion_params(
            LLMInferenceFramework.TEXT_GENERATION_INFERENCE, completion_sync_request
        )
    completion_sync_request.include_stop_str_in_output = None

    completion_sync_request.guided_regex = ""
    completion_sync_request.guided_json = {}
    completion_sync_request.guided_choice = [""]
    completion_sync_request.guided_grammar = ""
    with pytest.raises(ObjectHasInvalidValueException):
        validate_and_update_completion_params(LLMInferenceFramework.VLLM, completion_sync_request)

    completion_sync_request.guided_regex = None
    completion_sync_request.guided_choice = None
    completion_sync_request.guided_grammar = None
    with pytest.raises(ObjectHasInvalidValueException):
        validate_and_update_completion_params(
            LLMInferenceFramework.TEXT_GENERATION_INFERENCE, completion_sync_request
        )


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.domain.use_cases.llm_model_endpoint_use_cases.count_tokens",
    return_value=7,
)
async def test_completion_stream_use_case_success(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    fake_tokenizer_repository,
    llm_model_endpoint_streaming: ModelEndpoint,
    completion_stream_request: CompletionStreamV1Request,
):
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_streaming)
    fake_model_endpoint_service.streaming_model_endpoint_inference_gateway.responses = [
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"token": "I"}},
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"token": " am"}},
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"token": " a"}},
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"token": " new"}},
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"token": "bie"}},
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"token": "."}},
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={
                "result": {
                    "response": [
                        {
                            "error": None,
                            "text": "I am a newbie.",
                            "token_probs": {
                                "tokens": [
                                    "I",
                                    " am",
                                    " a",
                                    " new",
                                    "bie",
                                    ".",
                                ]
                            },
                            "tokens_consumed": 25,
                        }
                    ]
                }
            },
            traceback=None,
        ),
    ]
    use_case = CompletionStreamV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
        tokenizer_repository=fake_tokenizer_repository,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = use_case.execute(
        user=user,
        model_endpoint_name=llm_model_endpoint_streaming.record.name,
        request=completion_stream_request,
    )
    output_texts = ["I", " am", " a", " new", "bie", ".", "I am a newbie."]
    i = 0
    async for message in response_1:
        assert message.dict()["output"]["text"] == output_texts[i]
        if i == 6:
            assert message.dict()["output"]["num_prompt_tokens"] == 7
            assert message.dict()["output"]["num_completion_tokens"] == 6
        i += 1


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.domain.use_cases.llm_model_endpoint_use_cases.count_tokens",
    return_value=7,
)
async def test_completion_stream_vllm_use_case_success(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    fake_tokenizer_repository,
    llm_model_endpoint_stream: Tuple[ModelEndpoint, Any],
    completion_stream_request: CompletionStreamV1Request,
):
    completion_stream_request.guided_json = {}
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_stream[0])
    fake_model_endpoint_service.streaming_model_endpoint_inference_gateway.responses = [
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={
                "result": {
                    "text": "I",
                    "finished": False,
                    "count_prompt_tokens": 7,
                    "count_output_tokens": 1,
                }
            },
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={
                "result": {
                    "text": " am",
                    "finished": False,
                    "count_prompt_tokens": 7,
                    "count_output_tokens": 2,
                }
            },
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={
                "result": {
                    "text": " a",
                    "finished": False,
                    "count_prompt_tokens": 7,
                    "count_output_tokens": 3,
                }
            },
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={
                "result": {
                    "text": " new",
                    "finished": False,
                    "count_prompt_tokens": 7,
                    "count_output_tokens": 4,
                }
            },
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={
                "result": {
                    "text": "bie",
                    "finished": False,
                    "count_prompt_tokens": 7,
                    "count_output_tokens": 5,
                }
            },
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={
                "result": {
                    "text": ".",
                    "finished": True,
                    "count_prompt_tokens": 7,
                    "count_output_tokens": 6,
                }
            },
            traceback=None,
        ),
    ]
    use_case = CompletionStreamV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
        tokenizer_repository=fake_tokenizer_repository,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = use_case.execute(
        user=user,
        model_endpoint_name=llm_model_endpoint_stream[0].record.name,
        request=completion_stream_request,
    )
    output_texts = ["I", " am", " a", " new", "bie", ".", "I am a newbie."]
    i = 0
    async for message in response_1:
        assert message.dict()["output"]["text"] == output_texts[i]
        if i == 5:
            assert message.dict()["output"]["num_prompt_tokens"] == 7
            assert message.dict()["output"]["num_completion_tokens"] == 6
        i += 1


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.domain.use_cases.llm_model_endpoint_use_cases.count_tokens",
    return_value=7,
)
async def test_completion_stream_text_generation_inference_use_case_success(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    fake_tokenizer_repository,
    llm_model_endpoint_text_generation_inference: ModelEndpoint,
    completion_stream_request: CompletionStreamV1Request,
):
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_text_generation_inference)
    fake_model_endpoint_service.streaming_model_endpoint_inference_gateway.responses = [
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"token": {"text": "I"}}},
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"token": {"text": " am"}}},
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"token": {"text": " a"}}},
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"token": {"text": " new"}}},
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"token": {"text": "bie"}}},
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"token": {"text": "."}, "generated_text": "I am a newbie."}},
            traceback=None,
        ),
    ]
    use_case = CompletionStreamV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
        tokenizer_repository=fake_tokenizer_repository,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = use_case.execute(
        user=user,
        model_endpoint_name=llm_model_endpoint_text_generation_inference.record.name,
        request=completion_stream_request,
    )
    output_texts = ["I", " am", " a", " new", "bie", ".", "I am a newbie."]
    i = 0
    async for message in response_1:
        assert message.dict()["output"]["text"] == output_texts[i]
        if i == 5:
            assert message.dict()["output"]["num_prompt_tokens"] == 7
            assert message.dict()["output"]["num_completion_tokens"] == 6
        i += 1


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.domain.use_cases.llm_model_endpoint_use_cases.count_tokens",
    return_value=7,
)
async def test_completion_stream_trt_llm_use_case_success(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    fake_tokenizer_repository,
    llm_model_endpoint_trt_llm: ModelEndpoint,
    completion_stream_request: CompletionStreamV1Request,
):
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_trt_llm)
    fake_model_endpoint_service.streaming_model_endpoint_inference_gateway.responses = [
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"text_output": "Machine", "token_ids": 6189}},
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"text_output": "learning", "token_ids": 6509}},
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"text_output": "is", "token_ids": 338}},
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"text_output": "a", "token_ids": 263}},
            traceback=None,
        ),
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={"result": {"text_output": "branch", "token_ids": 5443}},
            traceback=None,
        ),
    ]
    use_case = CompletionStreamV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
        tokenizer_repository=fake_tokenizer_repository,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = use_case.execute(
        user=user,
        model_endpoint_name=llm_model_endpoint_trt_llm.record.name,
        request=completion_stream_request,
    )
    output_texts = ["Machine", "learning", "is", "a", "branch"]
    i = 0
    async for message in response_1:
        assert message.dict()["output"]["text"] == output_texts[i]
        assert message.dict()["output"]["num_prompt_tokens"] == 7
        assert message.dict()["output"]["num_completion_tokens"] == i + 1
        i += 1


@pytest.mark.asyncio
async def test_create_llm_fine_tune_model_name_valid():
    assert is_model_name_suffix_valid("model-name")
    assert not is_model_name_suffix_valid("Hi There! This is an invalid model name.")


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.domain.use_cases.llm_fine_tuning_use_cases.smart_open.open",
    mock.mock_open(read_data="prompt,response"),
)
async def test_create_fine_tune_success(
    fake_llm_fine_tuning_service,
    fake_model_endpoint_service,
    fake_llm_fine_tuning_events_repository,
    fake_file_storage_gateway,
    test_api_key: str,
):
    use_case = CreateFineTuneV1UseCase(
        fake_llm_fine_tuning_service,
        fake_model_endpoint_service,
        fake_llm_fine_tuning_events_repository,
        fake_file_storage_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    request = CreateFineTuneRequest(
        model="base_model",
        training_file="file1",
        validation_file=None,
        # fine_tuning_method="lora",
        hyperparameters={},
        suffix=None,
    )
    response = await use_case.execute(user=user, request=request)
    assert response.id

    # This erroring code is part of the service anyways
    # request.suffix = "Invalid model suffix *&^&%^$^%&^*"
    # with pytest.raises(InvalidRequestException):
    #     await use_case.execute(user=user, request=request)


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.domain.use_cases.llm_fine_tuning_use_cases.smart_open.open",
    mock.mock_open(read_data="prompt,response"),
)
async def test_create_fine_tune_limit(
    fake_llm_fine_tuning_service,
    fake_model_endpoint_service,
    fake_llm_fine_tuning_events_repository,
    fake_file_storage_gateway,
    test_api_key: str,
):
    use_case = CreateFineTuneV1UseCase(
        fake_llm_fine_tuning_service,
        fake_model_endpoint_service,
        fake_llm_fine_tuning_events_repository,
        fake_file_storage_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=False)
    request = CreateFineTuneRequest(
        model="base_model",
        training_file="file1",
        validation_file=None,
        # fine_tuning_method="lora",
        hyperparameters={},
        suffix=None,
    )
    for i in range(MAX_LLM_ENDPOINTS_PER_EXTERNAL_USER):
        if i == MAX_LLM_ENDPOINTS_PER_EXTERNAL_USER:
            with pytest.raises(LLMFineTuningQuotaReached):
                await use_case.execute(user=user, request=request)
        else:
            response = await use_case.execute(user=user, request=request)
            assert response.id


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.domain.use_cases.llm_fine_tuning_use_cases.smart_open.open",
    mock.mock_open(read_data="prompt,response"),
)
async def test_create_fine_tune_long_suffix(
    fake_llm_fine_tuning_service,
    fake_model_endpoint_service,
    fake_llm_fine_tuning_events_repository,
    fake_file_storage_gateway,
    test_api_key: str,
):
    use_case = CreateFineTuneV1UseCase(
        fake_llm_fine_tuning_service,
        fake_model_endpoint_service,
        fake_llm_fine_tuning_events_repository,
        fake_file_storage_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    request = CreateFineTuneRequest(
        model="base_model",
        training_file="file1",
        validation_file=None,
        # fine_tuning_method="lora",
        hyperparameters={},
        suffix="a" * 100,
    )
    with pytest.raises(InvalidRequestException):
        await use_case.execute(user=user, request=request)


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.domain.use_cases.llm_fine_tuning_use_cases.smart_open.open",
    mock.mock_open(read_data="prompt,not_response"),
)
async def test_create_fine_tune_invalid_headers(
    fake_llm_fine_tuning_service,
    fake_model_endpoint_service,
    fake_llm_fine_tuning_events_repository,
    fake_file_storage_gateway,
    test_api_key: str,
):
    use_case = CreateFineTuneV1UseCase(
        fake_llm_fine_tuning_service,
        fake_model_endpoint_service,
        fake_llm_fine_tuning_events_repository,
        fake_file_storage_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    request = CreateFineTuneRequest(
        model="base_model",
        training_file="file1",
        validation_file=None,
        # fine_tuning_method="lora",
        hyperparameters={},
        suffix=None,
    )
    with pytest.raises(InvalidRequestException):
        await use_case.execute(user=user, request=request)


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.domain.use_cases.llm_fine_tuning_use_cases.smart_open.open",
    mock.mock_open(read_data="prompt,response"),
)
async def test_get_fine_tune_events_success(
    fake_llm_fine_tuning_service,
    fake_llm_fine_tuning_events_repository,
    fake_model_endpoint_service,
    fake_file_storage_gateway,
    test_api_key: str,
):
    populate_use_case = CreateFineTuneV1UseCase(
        fake_llm_fine_tuning_service,
        fake_model_endpoint_service,
        fake_llm_fine_tuning_events_repository,
        fake_file_storage_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    request = CreateFineTuneRequest(
        model="base_model",
        training_file="file1",
        validation_file=None,
        # fine_tuning_method="lora",
        hyperparameters={},
        suffix=None,
    )
    response = await populate_use_case.execute(user=user, request=request)

    use_case = GetFineTuneEventsV1UseCase(
        llm_fine_tune_events_repository=fake_llm_fine_tuning_events_repository,
        llm_fine_tuning_service=fake_llm_fine_tuning_service,
    )
    response_2 = await use_case.execute(user=user, fine_tune_id=response.id)
    assert len(response_2.events) == len(fake_llm_fine_tuning_events_repository.all_events_list)


@pytest.mark.asyncio
async def test_download_model_success(
    fake_model_endpoint_service,
    fake_filesystem_gateway,
    fake_llm_artifact_gateway,
    model_endpoint_1: ModelEndpoint,
    test_api_key: str,
):
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    model_endpoint_1.record.owner = test_api_key
    model_endpoint_1.record.name = "base_model"
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    fake_llm_artifact_gateway._add_model(user.team_id, model_endpoint_1.record.name)
    use_case = ModelDownloadV1UseCase(
        fake_filesystem_gateway, fake_model_endpoint_service, fake_llm_artifact_gateway
    )
    request = ModelDownloadRequest(
        model_name=model_endpoint_1.record.name,
        download_format="huggingface",
    )
    response = await use_case.execute(user=user, request=request)
    assert response.urls != {}


@pytest.mark.asyncio
async def test_download_nonexistent_model_raises_not_found(
    fake_model_endpoint_service,
    fake_filesystem_gateway,
    fake_llm_artifact_gateway,
    model_endpoint_1: ModelEndpoint,
    test_api_key: str,
):
    model_endpoint_1.record.owner = test_api_key
    model_endpoint_1.record.name = "base_model"
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    fake_llm_artifact_gateway._add_model(test_api_key, "base_model")
    use_case = ModelDownloadV1UseCase(
        fake_filesystem_gateway, fake_model_endpoint_service, fake_llm_artifact_gateway
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    request = ModelDownloadRequest(
        model_name="nonexistent_model",
        download_format="huggingface",
    )
    with pytest.raises(ObjectNotFoundException):
        await use_case.execute(user=user, request=request)


@pytest.mark.asyncio
async def test_delete_model_success(
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    llm_model_endpoint_sync: Tuple[ModelEndpoint, Any],
    test_api_key: str,
):
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_sync[0])
    fake_model_endpoint_service.add_model_endpoint(llm_model_endpoint_sync[0])
    use_case = DeleteLLMEndpointByNameUseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response = await use_case.execute(
        user=user, model_endpoint_name=llm_model_endpoint_sync[0].record.name
    )
    remaining_endpoint_model_service = await fake_model_endpoint_service.get_model_endpoint(
        llm_model_endpoint_sync[0].record.id
    )
    assert remaining_endpoint_model_service is None
    assert response.deleted is True


@pytest.mark.asyncio
async def test_delete_nonexistent_model_raises_not_found(
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    llm_model_endpoint_sync: Tuple[ModelEndpoint, Any],
    test_api_key: str,
):
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_sync[0])
    fake_model_endpoint_service.add_model_endpoint(llm_model_endpoint_sync[0])
    use_case = DeleteLLMEndpointByNameUseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    with pytest.raises(ObjectNotFoundException):
        await use_case.execute(user=user, model_endpoint_name="nonexistent-model")


@pytest.mark.asyncio
async def test_delete_unauthorized_model_raises_not_authorized(
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    llm_model_endpoint_sync: Tuple[ModelEndpoint, Any],
):
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_sync[0])
    fake_model_endpoint_service.add_model_endpoint(llm_model_endpoint_sync[0])
    use_case = DeleteLLMEndpointByNameUseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
    )
    user = User(user_id="fakeapikey", team_id="fakeapikey", is_privileged_user=True)
    with pytest.raises(ObjectNotAuthorizedException):
        await use_case.execute(
            user=user, model_endpoint_name=llm_model_endpoint_sync[0].record.name
        )


@pytest.mark.asyncio
async def test_delete_public_inference_model_raises_not_authorized(
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    llm_model_endpoint_sync: Tuple[ModelEndpoint, Any],
    test_api_key,
):
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_sync[0])
    fake_model_endpoint_service.add_model_endpoint(llm_model_endpoint_sync[0])
    use_case = DeleteLLMEndpointByNameUseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
    )
    user = User(
        user_id="fakeapikey", team_id="faketeam", is_privileged_user=True
    )  # write access is based on team_id, so team_id != owner's team_id
    with pytest.raises(
        ObjectNotAuthorizedException
    ):  # user cannot delete public inference model they don't own
        await use_case.execute(
            user=user, model_endpoint_name=llm_model_endpoint_sync[0].record.name
        )


@pytest.mark.asyncio
async def test_validate_checkpoint_files_no_safetensors():
    fake_model_files = ["model-fake.bin", "model.json", "optimizer.pt"]
    with pytest.raises(ObjectHasInvalidValueException):
        validate_checkpoint_files(fake_model_files)


@pytest.mark.asyncio
async def test_validate_checkpoint_files_safetensors_with_other_files():
    fake_model_files = ["model-fake.bin", "model-fake2.safetensors", "model.json", "optimizer.pt"]
    validate_checkpoint_files(fake_model_files)  # No exception should be raised


def test_infer_hardware(fake_llm_artifact_gateway):
    fake_llm_artifact_gateway.model_config = {
        "architectures": ["MixtralForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 32768,
        "model_type": "mixtral",
        "num_attention_heads": 32,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "num_local_experts": 8,
        "rms_norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "router_aux_loss_coef": 0.02,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.36.0.dev0",
        "vocab_size": 32000,
    }
    hardware = _infer_hardware(fake_llm_artifact_gateway, "mixtral-8x7b", "")
    assert hardware.cpus == "20"
    assert hardware.gpus == 2
    assert hardware.memory == "160Gi"
    assert hardware.storage == "160Gi"
    assert hardware.gpu_type == GpuType.NVIDIA_HOPPER_H100

    fake_llm_artifact_gateway.model_config = {
        "architectures": ["MixtralForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 6144,
        "initializer_range": 0.02,
        "intermediate_size": 16384,
        "max_position_embeddings": 65536,
        "model_type": "mixtral",
        "num_attention_heads": 48,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 56,
        "num_key_value_heads": 8,
        "num_local_experts": 8,
        "rms_norm_eps": 1e-05,
        "rope_theta": 1000000,
        "router_aux_loss_coef": 0.001,
        "router_jitter_noise": 0.0,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0.dev0",
        "vocab_size": 32000,
    }
    hardware = _infer_hardware(fake_llm_artifact_gateway, "mixtral-8x22b", "")
    assert hardware.cpus == "80"
    assert hardware.gpus == 8
    assert hardware.memory == "800Gi"
    assert hardware.storage == "460Gi"
    assert hardware.gpu_type == GpuType.NVIDIA_HOPPER_H100

    fake_llm_artifact_gateway.model_config = {
        "_name_or_path": "meta-llama/Llama-2-7b-hf",
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 4096,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "torch_dtype": "float16",
        "transformers_version": "4.31.0.dev0",
        "vocab_size": 32000,
    }
    hardware = _infer_hardware(fake_llm_artifact_gateway, "llama-2-7b", "")
    assert hardware.cpus == "10"
    assert hardware.gpus == 1
    assert hardware.memory == "24Gi"
    assert hardware.storage == "80Gi"
    assert hardware.gpu_type == GpuType.NVIDIA_AMPERE_A10

    fake_llm_artifact_gateway.model_config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 8192,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_theta": 500000.0,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0.dev0",
        "vocab_size": 128256,
    }
    hardware = _infer_hardware(fake_llm_artifact_gateway, "llama-3-8b", "")
    assert hardware.cpus == "10"
    assert hardware.gpus == 1
    assert hardware.memory == "24Gi"
    assert hardware.storage == "80Gi"
    assert hardware.gpu_type == GpuType.NVIDIA_AMPERE_A10

    fake_llm_artifact_gateway.model_config = {
        "_name_or_path": "meta-llama/Llama-2-13b-hf",
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 13824,
        "max_position_embeddings": 4096,
        "model_type": "llama",
        "num_attention_heads": 40,
        "num_hidden_layers": 40,
        "num_key_value_heads": 40,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "torch_dtype": "float16",
        "transformers_version": "4.32.0.dev0",
        "vocab_size": 32000,
    }
    hardware = _infer_hardware(fake_llm_artifact_gateway, "llama-2-13b", "")
    assert hardware.cpus == "20"
    assert hardware.gpus == 2
    assert hardware.memory == "48Gi"
    assert hardware.storage == "80Gi"
    assert hardware.gpu_type == GpuType.NVIDIA_AMPERE_A10

    fake_llm_artifact_gateway.model_config = {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 8192,
        "initializer_range": 0.02,
        "intermediate_size": 22016,
        "max_position_embeddings": 16384,
        "model_type": "llama",
        "num_attention_heads": 64,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_theta": 1000000,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.32.0.dev0",
        "vocab_size": 32000,
    }
    hardware = _infer_hardware(fake_llm_artifact_gateway, "codellama-34b", "")
    assert hardware.cpus == "40"
    assert hardware.gpus == 4
    assert hardware.memory == "96Gi"
    assert hardware.storage == "96Gi"
    assert hardware.gpu_type == GpuType.NVIDIA_AMPERE_A10

    fake_llm_artifact_gateway.model_config = {
        "_name_or_path": "meta-llama/Llama-2-70b-hf",
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 8192,
        "initializer_range": 0.02,
        "intermediate_size": 28672,
        "max_position_embeddings": 4096,
        "model_type": "llama",
        "num_attention_heads": 64,
        "num_hidden_layers": 80,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "torch_dtype": "float16",
        "transformers_version": "4.32.0.dev0",
        "vocab_size": 32000,
    }
    hardware = _infer_hardware(fake_llm_artifact_gateway, "llama-2-70b", "")
    assert hardware.cpus == "20"
    assert hardware.gpus == 2
    assert hardware.memory == "160Gi"
    assert hardware.storage == "160Gi"
    assert hardware.gpu_type == GpuType.NVIDIA_HOPPER_H100

    fake_llm_artifact_gateway.model_config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "hidden_act": "silu",
        "hidden_size": 8192,
        "initializer_range": 0.02,
        "intermediate_size": 28672,
        "max_position_embeddings": 8192,
        "model_type": "llama",
        "num_attention_heads": 64,
        "num_hidden_layers": 80,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_theta": 500000.0,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0.dev0",
        "vocab_size": 128256,
    }
    hardware = _infer_hardware(fake_llm_artifact_gateway, "llama-3-70b", "")
    assert hardware.cpus == "20"
    assert hardware.gpus == 2
    assert hardware.memory == "160Gi"
    assert hardware.storage == "160Gi"
    assert hardware.gpu_type == GpuType.NVIDIA_HOPPER_H100

    hardware = _infer_hardware(fake_llm_artifact_gateway, "llama-3-70b", "", is_batch_job=True)
    assert hardware.cpus == "40"
    assert hardware.gpus == 4
    assert hardware.memory == "320Gi"
    assert hardware.storage == "320Gi"
    assert hardware.gpu_type == GpuType.NVIDIA_HOPPER_H100

    # (TODO) figure out how to calculate memory for llama-3-8b-instruct-262k
    # fake_llm_artifact_gateway.model_config = {
    #     "_name_or_path": "gradientai/llama3-8b-stage65k-chat",
    #     "architectures": ["LlamaForCausalLM"],
    #     "attention_dropout": 0.0,
    #     "bos_token_id": 128000,
    #     "eos_token_id": 128001,
    #     "hidden_act": "silu",
    #     "hidden_size": 4096,
    #     "initializer_range": 0.02,
    #     "intermediate_size": 14336,
    #     "max_position_embeddings": 262144,
    #     "model_type": "llama",
    #     "num_attention_heads": 32,
    #     "num_hidden_layers": 32,
    #     "num_key_value_heads": 8,
    #     "pretraining_tp": 1,
    #     "rms_norm_eps": 1e-05,
    #     "rope_theta": 283461213.0,
    #     "torch_dtype": "bfloat16",
    #     "transformers_version": "4.41.0.dev0",
    #     "vocab_size": 128256,
    # }
    # hardware = _infer_hardware(fake_llm_artifact_gateway, "llama-3-8b-instruct-262k", "")
    # assert hardware.cpus == "20"
    # assert hardware.gpus == 2
    # assert hardware.memory == "160Gi"
    # assert hardware.storage == "160Gi"
    # assert hardware.gpu_type == GpuType.NVIDIA_HOPPER_H100

    with pytest.raises(ObjectHasInvalidValueException):
        _infer_hardware(fake_llm_artifact_gateway, "unsupported_model", "")


def test_fill_hardware_info(fake_llm_artifact_gateway):
    request = CreateLLMModelEndpointV1Request(
        name="mixtral-8x7b",
        model_name="mixtral-8x7b",
        checkpoint_path="s3://checkpoint",
        metadata={},
        min_workers=1,
        max_workers=1,
        per_worker=1,
        labels={},
    )
    _fill_hardware_info(fake_llm_artifact_gateway, request)
    assert request.cpus == "20"
    assert request.gpus == 2
    assert request.memory == "160Gi"
    assert request.storage == "160Gi"
    assert request.gpu_type == GpuType.NVIDIA_HOPPER_H100

    request = CreateLLMModelEndpointV1Request(
        name="mixtral-8x7b",
        model_name="mixtral-8x7b",
        checkpoint_path="s3://checkpoint",
        metadata={},
        min_workers=1,
        max_workers=1,
        per_worker=1,
        labels={},
        gpus=1,
    )

    with pytest.raises(ObjectHasInvalidValueException):
        _fill_hardware_info(fake_llm_artifact_gateway, request)


@pytest.mark.asyncio
async def test_create_batch_completions(
    fake_docker_image_batch_job_gateway,
    fake_docker_repository_image_always_exists,
    fake_docker_image_batch_job_bundle_repository,
    fake_llm_artifact_gateway,
    test_api_key: str,
    create_batch_completions_request: CreateBatchCompletionsRequest,
):
    use_case = CreateBatchCompletionsUseCase(
        docker_image_batch_job_gateway=fake_docker_image_batch_job_gateway,
        docker_repository=fake_docker_repository_image_always_exists,
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository,
        llm_artifact_gateway=fake_llm_artifact_gateway,
    )

    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    result = await use_case.execute(user, create_batch_completions_request)

    job = await fake_docker_image_batch_job_gateway.get_docker_image_batch_job(result.job_id)
    assert job.num_workers == create_batch_completions_request.data_parallelism

    bundle = list(fake_docker_image_batch_job_bundle_repository.db.values())[0]
    assert bundle.command == [
        "dumb-init",
        "--",
        "/bin/bash",
        "-c",
        "ddtrace-run python vllm_batch.py",
    ]
