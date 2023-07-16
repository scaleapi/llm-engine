from typing import Any
from unittest.mock import Mock

import pytest

from llm_engine_server.domain.entities import ModelEndpoint
from llm_engine_server.infra.gateways import (
    LiveModelEndpointInfraGateway,
    live_model_endpoint_infra_gateway,
)


@pytest.fixture
def model_endpoint_infra_gateway(
    fake_resource_gateway, fake_task_queue_gateway
) -> LiveModelEndpointInfraGateway:
    return LiveModelEndpointInfraGateway(
        resource_gateway=fake_resource_gateway,
        task_queue_gateway=fake_task_queue_gateway,
    )


def test_create_model_endpoint_infra(
    model_endpoint_infra_gateway: LiveModelEndpointInfraGateway,
    model_endpoint_1: ModelEndpoint,
    model_endpoint_2: ModelEndpoint,
):
    for endpoint in [model_endpoint_1, model_endpoint_2]:
        assert endpoint.infra_state is not None
        prewarm = endpoint.infra_state.prewarm
        if prewarm is None:
            prewarm = True
        high_priority = endpoint.infra_state.high_priority
        if high_priority is None:
            high_priority = False
        endpoint_config = endpoint.infra_state.user_config_state.endpoint_config
        creation_task_id = model_endpoint_infra_gateway.create_model_endpoint_infra(
            model_endpoint_record=endpoint.record,
            min_workers=endpoint.infra_state.deployment_state.min_workers,
            max_workers=endpoint.infra_state.deployment_state.max_workers,
            per_worker=endpoint.infra_state.deployment_state.per_worker,
            cpus=endpoint.infra_state.resource_state.cpus,
            gpus=endpoint.infra_state.resource_state.gpus,
            memory=endpoint.infra_state.resource_state.memory,
            gpu_type=endpoint.infra_state.resource_state.gpu_type,
            storage=endpoint.infra_state.resource_state.storage,
            optimize_costs=bool(endpoint.infra_state.resource_state.optimize_costs),
            aws_role=endpoint.infra_state.aws_role,
            results_s3_bucket=endpoint.infra_state.results_s3_bucket,
            child_fn_info=endpoint.infra_state.child_fn_info,
            post_inference_hooks=(
                None if endpoint_config is None else endpoint_config.post_inference_hooks
            ),
            labels=endpoint.infra_state.labels,
            prewarm=prewarm,
            high_priority=high_priority,
            default_callback_url=(
                None if endpoint_config is None else endpoint_config.default_callback_url
            ),
            default_callback_auth=(
                None if endpoint_config is None else endpoint_config.default_callback_auth
            ),
        )
        assert creation_task_id


@pytest.mark.asyncio
async def test_update_model_endpoint_infra(
    model_endpoint_infra_gateway: LiveModelEndpointInfraGateway,
    model_endpoint_1: ModelEndpoint,
    model_endpoint_2: ModelEndpoint,
    fake_task_queue_gateway,
):
    resource_gateway: Any = model_endpoint_infra_gateway.resource_gateway
    existing_infra_state = model_endpoint_1.infra_state
    assert existing_infra_state is not None
    live_model_endpoint_infra_gateway.generate_deployment_name = Mock(
        return_value=existing_infra_state.deployment_name
    )
    resource_gateway.add_resource(model_endpoint_1.record.id, existing_infra_state)

    assert model_endpoint_2.infra_state is not None
    endpoint_config = model_endpoint_2.infra_state.user_config_state.endpoint_config
    creation_task_id_1 = await model_endpoint_infra_gateway.update_model_endpoint_infra(
        model_endpoint_record=model_endpoint_1.record,
        max_workers=model_endpoint_2.infra_state.deployment_state.max_workers,
        cpus=model_endpoint_2.infra_state.resource_state.cpus,
        memory=model_endpoint_2.infra_state.resource_state.memory,
        storage=model_endpoint_2.infra_state.resource_state.storage,
        post_inference_hooks=(
            None if endpoint_config is None else endpoint_config.post_inference_hooks
        ),
    )
    assert creation_task_id_1

    creation_task_id_2 = await model_endpoint_infra_gateway.update_model_endpoint_infra(
        model_endpoint_record=model_endpoint_1.record,
        min_workers=model_endpoint_2.infra_state.deployment_state.min_workers,
        per_worker=model_endpoint_2.infra_state.deployment_state.per_worker,
        gpus=model_endpoint_2.infra_state.resource_state.gpus,
        gpu_type=model_endpoint_2.infra_state.resource_state.gpu_type,
        child_fn_info=model_endpoint_2.infra_state.child_fn_info,
        labels=model_endpoint_2.infra_state.labels,
    )
    assert creation_task_id_2


@pytest.mark.asyncio
async def test_get_model_endpoint_infra_success(
    model_endpoint_infra_gateway: LiveModelEndpointInfraGateway,
    model_endpoint_1: ModelEndpoint,
):
    resource_gateway: Any = model_endpoint_infra_gateway.resource_gateway
    existing_infra_state = model_endpoint_1.infra_state
    assert existing_infra_state is not None
    live_model_endpoint_infra_gateway.generate_deployment_name = Mock(
        return_value=existing_infra_state.deployment_name
    )
    resource_gateway.add_resource(model_endpoint_1.record.id, existing_infra_state)
    model_endpoint_infra = await model_endpoint_infra_gateway.get_model_endpoint_infra(
        model_endpoint_record=model_endpoint_1.record
    )
    assert model_endpoint_infra == existing_infra_state


@pytest.mark.asyncio
async def test_get_model_endpoint_infra_returns_none(
    model_endpoint_infra_gateway: LiveModelEndpointInfraGateway,
    model_endpoint_1: ModelEndpoint,
):
    resource_gateway: Any = model_endpoint_infra_gateway.resource_gateway
    existing_infra_state = model_endpoint_1.infra_state
    assert existing_infra_state is not None

    resource_gateway.add_resource(
        endpoint_id=model_endpoint_1.record.id, infra_state=existing_infra_state
    )
    unknown_model_endpoint_record = model_endpoint_1.record.copy()
    unknown_model_endpoint_record.id = "other-id"
    model_endpoint_infra = await model_endpoint_infra_gateway.get_model_endpoint_infra(
        model_endpoint_record=unknown_model_endpoint_record
    )
    assert model_endpoint_infra is None


@pytest.mark.asyncio
async def test_delete_model_endpoint_infra(
    model_endpoint_infra_gateway: LiveModelEndpointInfraGateway,
    model_endpoint_1: ModelEndpoint,
):
    resource_gateway: Any = model_endpoint_infra_gateway.resource_gateway
    existing_infra_state = model_endpoint_1.infra_state
    assert existing_infra_state is not None
    live_model_endpoint_infra_gateway.generate_deployment_name = Mock(
        return_value=existing_infra_state.deployment_name
    )
    resource_gateway.add_resource(model_endpoint_1.record.id, existing_infra_state)
    successful = await model_endpoint_infra_gateway.delete_model_endpoint_infra(
        model_endpoint_record=model_endpoint_1.record
    )
    assert successful
