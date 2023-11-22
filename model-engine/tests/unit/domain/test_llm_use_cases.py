from typing import Any, List, Tuple
from unittest import mock

import pytest
from model_engine_server.common.dtos.llms import (
    CompletionOutput,
    CompletionStreamV1Request,
    CompletionSyncV1Request,
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
    MAX_LLM_ENDPOINTS_PER_INTERNAL_USER,
    CreateFineTuneV1UseCase,
    GetFineTuneEventsV1UseCase,
    is_model_name_suffix_valid,
)
from model_engine_server.domain.use_cases.llm_model_endpoint_use_cases import (
    CompletionStreamV1UseCase,
    CompletionSyncV1UseCase,
    CreateLLMModelBundleV1UseCase,
    CreateLLMModelEndpointV1UseCase,
    DeleteLLMEndpointByNameUseCase,
    GetLLMModelEndpointByNameV1UseCase,
    ModelDownloadV1UseCase,
    UpdateLLMModelEndpointV1UseCase,
    _include_safetensors_bin_or_pt,
)
from model_engine_server.domain.use_cases.model_bundle_use_cases import CreateModelBundleV2UseCase


@pytest.mark.asyncio
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
            "inference_framework_image_tag": create_llm_model_endpoint_request_async.inference_framework_image_tag,
            "num_shards": create_llm_model_endpoint_request_async.num_shards,
            "quantize": None,
            "checkpoint_path": None,
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
            "checkpoint_path": None,
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
            "checkpoint_path": None,
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
    )
    update_use_case = UpdateLLMModelEndpointV1UseCase(
        create_llm_model_bundle_use_case=llm_bundle_use_case,
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
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
            "inference_framework_image_tag": create_llm_model_endpoint_request_streaming.inference_framework_image_tag,
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
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_sync[0])
    fake_model_endpoint_service.sync_model_endpoint_inference_gateway.response = (
        SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={
                "result": [
                    {
                        "error": None,
                        "text": "I am a newbie to the world of programming.",
                        "token_probs": {
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
                            "token_probs": [
                                0.1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                            ],
                        },
                        "tokens_consumed": 25,
                    }
                ]
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
async def test_completion_sync_trt_llm_use_case_success(
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
    response_1 = await use_case.execute(
        user=user,
        model_endpoint_name=llm_model_endpoint_sync[0].record.name,
        request=completion_sync_request,
    )
    assert response_1.output is None


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
        assert message.dict()["request_id"]
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
        assert message.dict()["request_id"]
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
        assert message.dict()["request_id"]
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
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    request = CreateFineTuneRequest(
        model="base_model",
        training_file="file1",
        validation_file=None,
        # fine_tuning_method="lora",
        hyperparameters={},
        suffix=None,
    )
    for i in range(MAX_LLM_ENDPOINTS_PER_INTERNAL_USER):
        if i == MAX_LLM_ENDPOINTS_PER_INTERNAL_USER:
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
async def test_include_safetensors_bin_or_pt_majority_safetensors():
    fake_model_files = ["fake.bin", "fake2.safetensors", "model.json", "optimizer.pt"]
    assert _include_safetensors_bin_or_pt(fake_model_files) == "*.safetensors"


@pytest.mark.asyncio
async def test_include_safetensors_bin_or_pt_majority_bin():
    fake_model_files = [
        "fake.bin",
        "fake2.bin",
        "fake3.safetensors",
        "model.json",
        "optimizer.pt",
        "fake4.pt",
    ]
    assert _include_safetensors_bin_or_pt(fake_model_files) == "*.bin"


@pytest.mark.asyncio
async def test_include_safetensors_bin_or_pt_majority_pt():
    fake_model_files = [
        "fake.bin",
        "fake2.safetensors",
        "model.json",
        "optimizer.pt",
        "fake3.pt",
    ]
    assert _include_safetensors_bin_or_pt(fake_model_files) == "*.pt"
