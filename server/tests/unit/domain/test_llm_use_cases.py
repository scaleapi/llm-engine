from typing import Any, Tuple

import pytest
from llm_engine_server.common.dtos.llms import (
    CompletionOutput,
    CompletionStreamV1Request,
    CompletionSyncV1Request,
    CreateLLMModelEndpointV1Request,
    CreateLLMModelEndpointV1Response,
)
from llm_engine_server.common.dtos.tasks import SyncEndpointPredictV1Response, TaskStatus
from llm_engine_server.core.auth.authentication_repository import User
from llm_engine_server.core.domain_exceptions import (
    ObjectHasInvalidValueException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from llm_engine_server.domain.entities import ModelEndpoint, ModelEndpointType
from llm_engine_server.domain.exceptions import EndpointUnsupportedInferenceTypeException
from llm_engine_server.domain.use_cases.llm_model_endpoint_use_cases import (
    CompletionStreamV1UseCase,
    CompletionSyncV1UseCase,
    CreateLLMModelEndpointV1UseCase,
    GetLLMModelEndpointByNameV1UseCase,
)
from llm_engine_server.domain.use_cases.model_bundle_use_cases import CreateModelBundleV2UseCase


@pytest.mark.asyncio
async def test_create_model_endpoint_use_case_success(
    test_api_key: str,
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    fake_docker_repository_image_always_exists,
    fake_model_primitive_gateway,
    create_llm_model_endpoint_request_async: CreateLLMModelEndpointV1Request,
    create_llm_model_endpoint_request_sync: CreateLLMModelEndpointV1Request,
    create_llm_model_endpoint_request_streaming: CreateLLMModelEndpointV1Request,
):
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    bundle_use_case = CreateModelBundleV2UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    use_case = CreateLLMModelEndpointV1UseCase(
        create_model_bundle_use_case=bundle_use_case,
        model_bundle_repository=fake_model_bundle_repository,
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
        }
    }


@pytest.mark.asyncio
async def test_create_model_endpoint_text_generation_inference_use_case_success(
    test_api_key: str,
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    fake_docker_repository_image_always_exists,
    fake_model_primitive_gateway,
    create_llm_model_endpoint_text_generation_inference_request_async: CreateLLMModelEndpointV1Request,
    create_llm_model_endpoint_text_generation_inference_request_streaming: CreateLLMModelEndpointV1Request,
):
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    bundle_use_case = CreateModelBundleV2UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    use_case = CreateLLMModelEndpointV1UseCase(
        create_model_bundle_use_case=bundle_use_case,
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case.execute(
        user=user, request=create_llm_model_endpoint_text_generation_inference_request_streaming
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
        }
    }

    with pytest.raises(ObjectHasInvalidValueException):
        await use_case.execute(
            user=user, request=create_llm_model_endpoint_text_generation_inference_request_async
        )


@pytest.mark.asyncio
async def test_create_llm_model_endpoint_use_case_raises_invalid_value_exception(
    test_api_key: str,
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    fake_docker_repository_image_always_exists,
    fake_model_primitive_gateway,
    create_llm_model_endpoint_request_invalid_model_name: CreateLLMModelEndpointV1Request,
):
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    bundle_use_case = CreateModelBundleV2UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    use_case = CreateLLMModelEndpointV1UseCase(
        create_model_bundle_use_case=bundle_use_case,
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    with pytest.raises(ObjectHasInvalidValueException):
        await use_case.execute(
            user=user, request=create_llm_model_endpoint_request_invalid_model_name
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
async def test_completion_sync_use_case_success(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
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
                            ]
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
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case.execute(
        user=user,
        model_endpoint_name=llm_model_endpoint_sync[0].record.name,
        request=completion_sync_request,
    )
    assert response_1.status == TaskStatus.SUCCESS
    assert response_1.outputs == [
        CompletionOutput(
            text="I am a newbie to the world of programming.",
            num_prompt_tokens=14,
            num_completion_tokens=11,
        )
    ]


@pytest.mark.asyncio
async def test_completion_sync_text_generation_inference_use_case_success(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
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
          "text": " Deep"
        },
        {
          "text": " Learning"
        },
        {
          "text": " is"
        },
        {
          "text": " a"
        },
        {
          "text": " new"
        },
        {
          "text": " type"
        },
        {
          "text": " of"
        },
        {
          "text": " machine"
        },
        {
          "text": " learning"
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
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case.execute(
        user=user,
        model_endpoint_name=llm_model_endpoint_text_generation_inference.record.name,
        request=completion_sync_request,
    )
    assert response_1.status == TaskStatus.SUCCESS
    print(response_1.outputs)
    assert response_1.outputs == [
        CompletionOutput(
            text=" Deep Learning is a new type of machine learning",
            num_prompt_tokens=None,
            num_completion_tokens=9,
        ),
        CompletionOutput(
            text=" Deep Learning is a new type of machine learning",
            num_prompt_tokens=None,
            num_completion_tokens=9,
        ),
    ]


@pytest.mark.asyncio
async def test_completion_sync_use_case_predict_failed(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
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
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case.execute(
        user=user,
        model_endpoint_name=llm_model_endpoint_sync[0].record.name,
        request=completion_sync_request,
    )
    assert response_1.status == TaskStatus.FAILURE
    assert len(response_1.outputs) == 0
    assert response_1.traceback == "failed to predict"


@pytest.mark.asyncio
async def test_completion_sync_use_case_not_sync_endpoint_raises(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
    llm_model_endpoint_async: Tuple[ModelEndpoint, Any],
    completion_sync_request: CompletionSyncV1Request,
):
    fake_llm_model_endpoint_service.add_model_endpoint(llm_model_endpoint_async[0])
    use_case = CompletionSyncV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
        llm_model_endpoint_service=fake_llm_model_endpoint_service,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    with pytest.raises(EndpointUnsupportedInferenceTypeException):
        await use_case.execute(
            user=user,
            model_endpoint_name=llm_model_endpoint_async[0].record.name,
            request=completion_sync_request,
        )


@pytest.mark.asyncio
async def test_completion_stream_use_case_success(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
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
        assert message.dict()["status"] == "SUCCESS"
        assert message.dict()["output"]["text"] == output_texts[i]
        if i == 6:
            assert message.dict()["output"]["num_prompt_tokens"] == 19
            assert message.dict()["output"]["num_completion_tokens"] == 6
        i += 1


@pytest.mark.asyncio
async def test_completion_stream_text_generation_inference_use_case_success(
    test_api_key: str,
    fake_model_endpoint_service,
    fake_llm_model_endpoint_service,
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
        assert message.dict()["status"] == "SUCCESS"
        assert message.dict()["output"]["text"] == output_texts[i]
        if i == 5:
            assert message.dict()["output"]["num_completion_tokens"] == 6
        i += 1
