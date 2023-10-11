import json
import re
from typing import Any, Dict, Tuple

import pytest
from model_engine_server.common.dtos.llms import GetLLMModelEndpointV1Response
from model_engine_server.domain.entities import ModelEndpoint


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
    )
    response_1 = client.post(
        f"/v1/llm/completions-sync?model_endpoint_name={llm_model_endpoint_sync[0].record.name}",
        auth=("no_user", ""),
        json=completion_sync_request,
    )
    assert response_1.status_code == 200
    assert response_1.json()["output"] is None
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


@pytest.mark.skip(reason="Need to figure out FastAPI test client asyncio funkiness")
def test_completion_stream_success(
    llm_model_endpoint_streaming: ModelEndpoint,
    completion_stream_request: Dict[str, Any],
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
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
    )
    response_1 = client.post(
        f"/v1/llm/completions-stream?model_endpoint_name={llm_model_endpoint_streaming.record.name}",
        auth=("no_user", ""),
        json=completion_stream_request,
        stream=True,
    )
    assert response_1.status_code == 200
    count = 0
    for message in response_1:
        assert re.fullmatch(
            'data: {"request_id"}: ".*", "output": null}\r\n\r\n',
            message.decode("utf-8"),
        )
        count += 1
    assert count == 1


@pytest.mark.skip(reason="Need to figure out FastAPI test client asyncio funkiness")
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
    response_1 = client.post(
        f"/v1/llm/completions-stream?model_endpoint_name={llm_model_endpoint_streaming.record.name}",
        auth=("no_user", ""),
        json=completion_stream_request,
        stream=True,
    )

    assert response_1.status_code == 200

    for message in response_1:
        assert "404" in message.decode("utf-8")
