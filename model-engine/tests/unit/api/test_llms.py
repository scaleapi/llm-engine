import json
from typing import Any, Dict, Tuple
from unittest import mock

import pytest
from model_engine_server.common.dtos.llms import GetLLMModelEndpointV1Response
from model_engine_server.common.dtos.tasks import SyncEndpointPredictV1Response, TaskStatus
from model_engine_server.domain.entities import ModelEndpoint

from ..conftest import mocked__get_recommended_hardware_config_map


def test_create_llm_model_endpoint_success(
    create_llm_model_endpoint_request_sync: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response_1 = client.post(
        "/v1/llm/model-endpoints",
        auth=(test_api_key, ""),
        json=create_llm_model_endpoint_request_sync,
    )
    assert response_1.status_code == 200


def test_list_model_endpoints_success(
    llm_model_endpoint_async: Tuple[ModelEndpoint, Any],
    model_endpoint_2: Tuple[ModelEndpoint, Any],
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_model_endpoint_record_repository_contents={
            llm_model_endpoint_async[0].record.id: llm_model_endpoint_async[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            llm_model_endpoint_async[0]
            .infra_state.deployment_name: llm_model_endpoint_async[0]
            .infra_state,
            model_endpoint_2[0].infra_state.deployment_name: model_endpoint_2[0].infra_state,
        },
    )
    response_1 = client.get(
        "/v1/llm/model-endpoints?order_by=newest",
        auth=("no_user", ""),
    )
    expected_model_endpoint_1 = json.loads(
        GetLLMModelEndpointV1Response.parse_obj(llm_model_endpoint_async[1]).json()
    )
    assert response_1.status_code == 200
    assert response_1.json() == {"model_endpoints": [expected_model_endpoint_1]}


def test_get_llm_model_endpoint_success(
    llm_model_endpoint_sync: Tuple[ModelEndpoint, Any],
    model_endpoint_2: Tuple[ModelEndpoint, Any],
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_model_endpoint_record_repository_contents={
            llm_model_endpoint_sync[0].record.id: llm_model_endpoint_sync[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            llm_model_endpoint_sync[0]
            .infra_state.deployment_name: llm_model_endpoint_sync[0]
            .infra_state,
            model_endpoint_2[0].infra_state.deployment_name: model_endpoint_2[0].infra_state,
        },
    )
    response_1 = client.get(
        f"/v1/llm/model-endpoints/{llm_model_endpoint_sync[0].record.name}",
        auth=("no_user", ""),
    )
    expected_model_endpoint_1 = json.loads(
        GetLLMModelEndpointV1Response.parse_obj(llm_model_endpoint_sync[1]).json()
    )
    assert response_1.status_code == 200
    assert response_1.json() == expected_model_endpoint_1


def test_completion_sync_success(
    llm_model_endpoint_sync: Tuple[ModelEndpoint, Any],
    completion_sync_request: Dict[str, Any],
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={
            llm_model_endpoint_sync[0].record.id: llm_model_endpoint_sync[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            llm_model_endpoint_sync[0]
            .infra_state.deployment_name: llm_model_endpoint_sync[0]
            .infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
        fake_sync_inference_content=SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS,
            result={
                "result": """{
                    "text": "output",
                    "count_prompt_tokens": 1,
                    "count_output_tokens": 1
                }"""
            },
            traceback=None,
        ),
    )
    response_1 = client.post(
        f"/v1/llm/completions-sync?model_endpoint_name={llm_model_endpoint_sync[0].record.name}",
        auth=("no_user", ""),
        json=completion_sync_request,
    )
    assert response_1.status_code == 200
    assert response_1.json()["output"] == {
        "text": "output",
        "num_completion_tokens": 1,
        "num_prompt_tokens": 1,
        "tokens": None,
    }
    assert response_1.json().keys() == {"output", "request_id"}


def test_completion_sync_endpoint_not_found_returns_404(
    llm_model_endpoint_sync: Tuple[ModelEndpoint, Any],
    completion_sync_request: Dict[str, Any],
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={
            llm_model_endpoint_sync[0]
            .infra_state.deployment_name: llm_model_endpoint_sync[0]
            .infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response_1 = client.post(
        f"/v1/llm/completions-sync?model_endpoint_name={llm_model_endpoint_sync[0].record.name}",
        auth=("no_user", ""),
        json=completion_sync_request,
    )
    assert response_1.status_code == 404


@pytest.mark.asyncio
async def test_completion_stream_success(
    llm_model_endpoint_streaming: ModelEndpoint,
    completion_stream_request: Dict[str, Any],
    get_async_test_client_wrapper,
):  # pragma: no cover
    async with get_async_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={
            llm_model_endpoint_streaming.record.id: llm_model_endpoint_streaming.record,
        },
        fake_model_endpoint_infra_gateway_contents={
            llm_model_endpoint_streaming.infra_state.deployment_name: llm_model_endpoint_streaming.infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    ) as client:
        with mock.patch(
            "model_engine_server.domain.use_cases.llm_model_endpoint_use_cases.count_tokens",
            return_value=5,
        ):
            async with client.stream(
                method="POST",
                url=f"/v1/llm/completions-stream?model_endpoint_name={llm_model_endpoint_streaming.record.name}",
                auth=("no_user", ""),
                json=completion_stream_request,
            ) as r:
                assert r.status_code == 200
                count = 0
                async for message in r.aiter_bytes():
                    decoded_message = message.decode("utf-8")
                    assert decoded_message.startswith(
                        "data: "
                    ), f"SSE does not start with 'data: ', message is '{decoded_message}'"

                    # strip 'data: ' prefix from  Server-sent events format
                    json_str = decoded_message[len("data: ") :]
                    parsed_data = json.loads(json_str.strip())
                    assert parsed_data["request_id"] is not None
                    assert parsed_data["output"] is None
                    assert parsed_data["error"] is None
                    count += 1
                assert count == 1


def test_completion_stream_endpoint_not_found_returns_404(
    llm_model_endpoint_streaming: ModelEndpoint,
    completion_stream_request: Dict[str, Any],
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={
            llm_model_endpoint_streaming.infra_state.deployment_name: llm_model_endpoint_streaming.infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    with client.stream(
        method="POST",
        url=f"/v1/llm/completions-stream?model_endpoint_name={llm_model_endpoint_streaming.record.name}",
        auth=("no_user", ""),
        json=completion_stream_request,
    ) as r:
        assert r.status_code == 404


def test_completion_stream_misc_server_error_returns_500(
    llm_model_endpoint_streaming: ModelEndpoint,
    completion_stream_request: Dict[str, Any],
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={
            llm_model_endpoint_streaming.infra_state.deployment_name: llm_model_endpoint_streaming.infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    with mock.patch(
        "model_engine_server.domain.use_cases.llm_model_endpoint_use_cases.CompletionStreamV1UseCase.execute",
    ) as mock_stream_usecase:
        mock_stream_usecase.side_effect = RuntimeError("Some server side runtime error.")
        with client.stream(
            method="POST",
            url=f"/v1/llm/completions-stream?model_endpoint_name={llm_model_endpoint_streaming.record.name}",
            auth=("no_user", ""),
            json=completion_stream_request,
        ) as r:
            assert r.status_code == 500


@mock.patch(
    "model_engine_server.domain.use_cases.llm_model_endpoint_use_cases._get_recommended_hardware_config_map",
    mocked__get_recommended_hardware_config_map(),
)
def test_create_batch_completions_success(
    create_batch_completions_request: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response_1 = client.post(
        "/v1/llm/batch-completions",
        auth=(test_api_key, ""),
        json=create_batch_completions_request,
    )
    assert response_1.status_code == 200
