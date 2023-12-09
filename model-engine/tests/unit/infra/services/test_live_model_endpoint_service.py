from typing import Any, List, Optional
from unittest.mock import AsyncMock

import pytest
from model_engine_server.domain.entities import (
    ModelBundle,
    ModelEndpoint,
    ModelEndpointRecord,
    ModelEndpointStatus,
    ShadowModelEndpointRecord,
)
from model_engine_server.domain.exceptions import (
    EndpointDeleteFailedException,
    ExistingEndpointOperationInProgressException,
    ObjectAlreadyExistsException,
    ObjectNotFoundException,
)
from model_engine_server.infra.services import LiveModelEndpointService


async def _create_model_endpoint_helper(
    model_endpoint: ModelEndpoint,
    service: LiveModelEndpointService,
    shadow_endpoints: Optional[List[ShadowModelEndpointRecord]] = None,
) -> ModelEndpointRecord:
    assert model_endpoint.infra_state is not None
    infra_state = model_endpoint.infra_state
    assert infra_state.user_config_state.endpoint_config is not None
    prewarm = infra_state.prewarm
    if prewarm is None:
        prewarm = True
    high_priority = infra_state.high_priority
    if high_priority is None:
        high_priority = False
    endpoint_config = infra_state.user_config_state.endpoint_config
    model_endpoint_record = await service.create_model_endpoint(
        name=model_endpoint.record.name,
        created_by=model_endpoint.record.created_by,
        model_bundle_id=model_endpoint.record.current_model_bundle.id,
        endpoint_type=model_endpoint.record.endpoint_type,
        metadata=model_endpoint.record.metadata,
        post_inference_hooks=(
            None if endpoint_config is None else endpoint_config.post_inference_hooks
        ),
        default_callback_url=(
            None if endpoint_config is None else endpoint_config.default_callback_url
        ),
        default_callback_auth=(
            None if endpoint_config is None else endpoint_config.default_callback_auth
        ),
        child_fn_info=infra_state.child_fn_info,
        cpus=infra_state.resource_state.cpus,
        gpus=infra_state.resource_state.gpus,
        memory=infra_state.resource_state.memory,
        gpu_type=infra_state.resource_state.gpu_type,
        storage=infra_state.resource_state.storage,
        optimize_costs=bool(infra_state.resource_state.optimize_costs),
        min_workers=infra_state.deployment_state.min_workers,
        max_workers=infra_state.deployment_state.max_workers,
        per_worker=infra_state.deployment_state.per_worker,
        labels=infra_state.labels,
        aws_role=infra_state.aws_role,
        results_s3_bucket=infra_state.results_s3_bucket,
        prewarm=prewarm,
        high_priority=high_priority,
        billing_tags=infra_state.user_config_state.endpoint_config.billing_tags,
        owner=model_endpoint.record.owner,
        shadow_endpoints=shadow_endpoints,
    )
    return model_endpoint_record


@pytest.mark.asyncio
async def test_create_and_update_model_endpoint_with_shadows_success(
    model_endpoint_1: ModelEndpoint,
    model_endpoint_2: ModelEndpoint,
    model_endpoint_3: ModelEndpoint,
    model_endpoint_4: ModelEndpoint,
    fake_live_model_endpoint_service: LiveModelEndpointService,
):

    # Create shadow endpoints
    shadow_model_endpoint_2_record = await _create_model_endpoint_helper(
        model_endpoint=model_endpoint_2,
        service=fake_live_model_endpoint_service,
    )
    shadow_model_endpoint_3_record = await _create_model_endpoint_helper(
        model_endpoint=model_endpoint_3,
        service=fake_live_model_endpoint_service,
    )
    shadow_model_endpoint_2 = ShadowModelEndpointRecord(id=shadow_model_endpoint_2_record.id)
    shadow_model_endpoint_3 = ShadowModelEndpointRecord(id=shadow_model_endpoint_3_record.id)

    # Create new model endpoint with shadow endpoints
    model_endpoint_record_with_shadows = await _create_model_endpoint_helper(
        model_endpoint=model_endpoint_1,
        service=fake_live_model_endpoint_service,
        shadow_endpoints=[shadow_model_endpoint_2, shadow_model_endpoint_3],
    )
    assert isinstance(model_endpoint_record_with_shadows, ModelEndpointRecord)
    assert model_endpoint_record_with_shadows.shadow_endpoints_ids == [
        shadow_model_endpoint_2_record.id,
        shadow_model_endpoint_3_record.id,
    ]

    # Create a new shadow endpoint
    shadow_model_endpoint_4_record = await _create_model_endpoint_helper(
        model_endpoint=model_endpoint_4,
        service=fake_live_model_endpoint_service,
    )
    shadow_model_endpoint_4 = ShadowModelEndpointRecord(id=shadow_model_endpoint_4_record.id)

    # Promote the endpoint infra for the new shadow endpoint
    model_endpoint_infra_gateway: Any = (
        fake_live_model_endpoint_service.model_endpoint_infra_gateway
    )
    await model_endpoint_infra_gateway.promote_in_flight_infra(
        owner=model_endpoint_record_with_shadows.created_by,
        model_endpoint_name=model_endpoint_record_with_shadows.name,
    )

    # Update the model endpoint with the new shadow endpoint
    update_kwargs: Any = dict(
        shadow_endpoints=[shadow_model_endpoint_4],
    )
    updated_model_endpoint_record_with_shadows = (
        await fake_live_model_endpoint_service.update_model_endpoint(
            model_endpoint_id=model_endpoint_record_with_shadows.id, **update_kwargs
        )
    )
    assert (updated_model_endpoint_record_with_shadows.shadow_endpoints_ids) == [
        shadow_model_endpoint_4_record.id
    ]


@pytest.mark.asyncio
async def test_create_get_model_endpoint_success(
    model_endpoint_1: ModelEndpoint,
    fake_live_model_endpoint_service: LiveModelEndpointService,
):
    # Create the model endpoint.
    model_endpoint_record = await _create_model_endpoint_helper(
        model_endpoint=model_endpoint_1, service=fake_live_model_endpoint_service
    )

    # Check that the endpoint records match.
    assert isinstance(model_endpoint_record, ModelEndpointRecord)
    assert model_endpoint_record.status == ModelEndpointStatus.UPDATE_PENDING
    attributes_to_check = {"name", "created_by", "endpoint_type", "metadata"}
    for attribute in attributes_to_check:
        actual_attribute = model_endpoint_record.__getattribute__(attribute)
        expected_attribute = model_endpoint_1.record.__getattribute__(attribute)
        assert actual_attribute == expected_attribute

    # Now promote the endpoint infra to ready and check that the states match.
    model_endpoint_infra_gateway: Any = (
        fake_live_model_endpoint_service.model_endpoint_infra_gateway
    )
    await model_endpoint_infra_gateway.promote_in_flight_infra(
        owner=model_endpoint_record.created_by,
        model_endpoint_name=model_endpoint_record.name,
    )
    model_endpoint = await fake_live_model_endpoint_service.get_model_endpoint(
        model_endpoint_record.id
    )
    assert model_endpoint is not None

    # The following features don't necessarily match. After fixing them, check for equality.
    assert model_endpoint.infra_state is not None
    assert model_endpoint_1.infra_state is not None
    model_endpoint.infra_state.deployment_state.available_workers = (
        model_endpoint_1.infra_state.deployment_state.available_workers
    )
    model_endpoint.infra_state.deployment_state.unavailable_workers = (
        model_endpoint_1.infra_state.deployment_state.unavailable_workers
    )
    model_endpoint.record.created_at = model_endpoint_1.record.created_at
    model_endpoint.record.last_updated_at = model_endpoint_1.record.last_updated_at
    model_endpoint.record.id = model_endpoint_1.record.id
    model_endpoint.infra_state.user_config_state.endpoint_config.billing_tags = model_endpoint_1.infra_state.user_config_state.endpoint_config.billing_tags  # type: ignore
    # Use dict comparison because errors are more readable.
    assert model_endpoint.dict() == model_endpoint_1.dict()


@pytest.mark.asyncio
async def test_create_model_endpoint_raises_already_exists(
    fake_live_model_endpoint_service: LiveModelEndpointService,
    model_endpoint_1: ModelEndpoint,
):
    model_endpoint_record_repository: Any = (
        fake_live_model_endpoint_service.model_endpoint_record_repository
    )
    model_endpoint_record_repository.add_model_endpoint_record(model_endpoint_1.record)
    assert model_endpoint_1.infra_state is not None
    infra_state = model_endpoint_1.infra_state
    prewarm = infra_state.prewarm
    if prewarm is None:
        prewarm = True
    high_priority = infra_state.high_priority
    if high_priority is None:
        high_priority = False
    endpoint_config = infra_state.user_config_state.endpoint_config
    with pytest.raises(ObjectAlreadyExistsException):
        await fake_live_model_endpoint_service.create_model_endpoint(
            name=model_endpoint_1.record.name,
            created_by=model_endpoint_1.record.created_by,
            model_bundle_id=model_endpoint_1.record.current_model_bundle.id,
            endpoint_type=model_endpoint_1.record.endpoint_type,
            metadata=model_endpoint_1.record.metadata,
            post_inference_hooks=(
                None if endpoint_config is None else endpoint_config.post_inference_hooks
            ),
            default_callback_url=(
                None if endpoint_config is None else endpoint_config.default_callback_url
            ),
            default_callback_auth=(
                None if endpoint_config is None else endpoint_config.default_callback_auth
            ),
            child_fn_info=infra_state.child_fn_info,
            cpus=infra_state.resource_state.cpus,
            gpus=infra_state.resource_state.gpus,
            memory=infra_state.resource_state.memory,
            gpu_type=infra_state.resource_state.gpu_type,
            storage=infra_state.resource_state.storage,
            optimize_costs=bool(infra_state.resource_state.optimize_costs),
            min_workers=infra_state.deployment_state.min_workers,
            max_workers=infra_state.deployment_state.max_workers,
            per_worker=infra_state.deployment_state.per_worker,
            labels=infra_state.labels,
            aws_role=infra_state.aws_role,
            results_s3_bucket=infra_state.results_s3_bucket,
            prewarm=prewarm,
            high_priority=high_priority,
            owner=model_endpoint_1.record.owner,
        )


@pytest.mark.asyncio
async def test_get_model_endpoint_returns_none(
    fake_live_model_endpoint_service: LiveModelEndpointService,
):
    model_endpoint = await fake_live_model_endpoint_service.get_model_endpoint(
        model_endpoint_id="invalid_model_endpoint_id"
    )
    assert model_endpoint is None


@pytest.mark.asyncio
async def test_create_update_model_endpoint_success(
    fake_live_model_endpoint_service: LiveModelEndpointService,
    model_bundle_2: ModelBundle,
    model_endpoint_1: ModelEndpoint,
):
    # Create the model endpoint.
    model_endpoint_record = await _create_model_endpoint_helper(
        model_endpoint=model_endpoint_1, service=fake_live_model_endpoint_service
    )

    # Now promote the endpoint infra to ready.
    model_endpoint_infra_gateway: Any = (
        fake_live_model_endpoint_service.model_endpoint_infra_gateway
    )
    await model_endpoint_infra_gateway.promote_in_flight_infra(
        owner=model_endpoint_record.created_by,
        model_endpoint_name=model_endpoint_record.name,
    )

    # Modify the creation task ID to check that it's reset to the test value.
    model_endpoint_record.creation_task_id = "some_other_creation_task_id"

    # Updating the model endpoint should be successful now.
    update_kwargs: Any = dict(
        model_bundle_id=model_bundle_2.id,
        metadata={"some_new_key": "some_new_values"},
        cpus=4,
        min_workers=2,
        max_workers=5,
        labels={"some_new_label_key": "some_new_label_value"},
    )
    updated_model_endpoint_record = await fake_live_model_endpoint_service.update_model_endpoint(
        model_endpoint_id=model_endpoint_record.id, **update_kwargs
    )
    assert updated_model_endpoint_record.current_model_bundle.id == update_kwargs["model_bundle_id"]
    assert updated_model_endpoint_record.metadata == update_kwargs["metadata"]
    assert updated_model_endpoint_record.status == ModelEndpointStatus.UPDATE_PENDING
    assert updated_model_endpoint_record.creation_task_id != "some_other_creation_task_id"

    # Now promote the endpoint infra to ready and check that the states match.
    await model_endpoint_infra_gateway.promote_in_flight_infra(
        owner=model_endpoint_record.created_by,
        model_endpoint_name=model_endpoint_record.name,
    )
    model_endpoint = await fake_live_model_endpoint_service.get_model_endpoint(
        model_endpoint_record.id
    )

    # Check that fields of the updated endpoint match.
    assert model_endpoint is not None
    assert model_endpoint.infra_state is not None
    assert model_endpoint.record.current_model_bundle.id == update_kwargs["model_bundle_id"]
    assert model_endpoint.record.metadata == update_kwargs["metadata"]
    assert model_endpoint.infra_state.resource_state.cpus == update_kwargs["cpus"]
    assert model_endpoint.infra_state.deployment_state.min_workers == update_kwargs["min_workers"]
    assert model_endpoint.infra_state.deployment_state.max_workers == update_kwargs["max_workers"]
    assert model_endpoint.infra_state.labels == update_kwargs["labels"]


@pytest.mark.skip(reason="Exception is temporarily disabled due to lock flakiness")
@pytest.mark.asyncio
async def test_create_update_model_endpoint_raises_existing_operation_in_progress(
    fake_live_model_endpoint_service: LiveModelEndpointService,
    model_endpoint_1: ModelEndpoint,
):
    # Create the model endpoint.
    model_endpoint_record = await _create_model_endpoint_helper(
        model_endpoint=model_endpoint_1, service=fake_live_model_endpoint_service
    )

    # Update the model endpoint's status to update in progress.
    await fake_live_model_endpoint_service.model_endpoint_record_repository.update_model_endpoint_record(
        model_endpoint_id=model_endpoint_record.id,
        status=ModelEndpointStatus.UPDATE_IN_PROGRESS,
    )

    # Updating the model endpoint before it's ready should raise an exception.
    update_kwargs: Any = dict(
        metadata={"some_new_key": "some_new_values"},
        cpus=4,
        min_workers=2,
        max_workers=5,
        labels={"some_new_label_key": "some_new_label_value"},
    )
    with pytest.raises(ExistingEndpointOperationInProgressException):
        await fake_live_model_endpoint_service.update_model_endpoint(
            model_endpoint_id=model_endpoint_record.id, **update_kwargs
        )


@pytest.mark.skip(reason="Exception is temporarily disabled due to lock flakiness")
@pytest.mark.asyncio
async def test_create_update_model_endpoint_lock_not_acquired_raises_existing_operation_in_progress(
    fake_live_model_endpoint_service: LiveModelEndpointService,
    model_endpoint_1: ModelEndpoint,
):
    # Create the model endpoint.
    model_endpoint_record = await _create_model_endpoint_helper(
        model_endpoint=model_endpoint_1, service=fake_live_model_endpoint_service
    )

    # Force lock the model endpoint.
    model_endpoint_record_repo: Any = (
        fake_live_model_endpoint_service.model_endpoint_record_repository
    )
    model_endpoint_record_repo.force_lock_model_endpoint(
        model_endpoint_record=model_endpoint_record
    )

    # Updating the model endpoint before it's ready should raise an exception.
    update_kwargs: Any = dict(
        metadata={"some_new_key": "some_new_values"},
        cpus=4,
        min_workers=2,
        max_workers=5,
        labels={"some_new_label_key": "some_new_label_value"},
    )
    with pytest.raises(ExistingEndpointOperationInProgressException):
        await fake_live_model_endpoint_service.update_model_endpoint(
            model_endpoint_id=model_endpoint_record.id, **update_kwargs
        )


@pytest.mark.asyncio
async def test_update_model_endpoint_raises_not_found(
    fake_live_model_endpoint_service: LiveModelEndpointService,
):
    with pytest.raises(ObjectNotFoundException):
        await fake_live_model_endpoint_service.update_model_endpoint(
            model_endpoint_id="invalid_model_endpoint_id", cpus=4
        )


@pytest.mark.asyncio
async def test_create_delete_model_endpoint_success(
    fake_live_model_endpoint_service: LiveModelEndpointService,
    model_bundle_2: ModelBundle,
    model_endpoint_1: ModelEndpoint,
):
    # Create the model endpoint.
    model_endpoint_record = await _create_model_endpoint_helper(
        model_endpoint=model_endpoint_1, service=fake_live_model_endpoint_service
    )

    # Now promote the endpoint infra to ready.
    model_endpoint_infra_gateway: Any = (
        fake_live_model_endpoint_service.model_endpoint_infra_gateway
    )
    await model_endpoint_infra_gateway.promote_in_flight_infra(
        owner=model_endpoint_record.created_by,
        model_endpoint_name=model_endpoint_record.name,
    )

    # Modify the creation task ID to check that it's reset to the test value.
    model_endpoint_record.creation_task_id = "some_other_creation_task_id"

    # Deleting the model endpoint should be successful now.
    await fake_live_model_endpoint_service.delete_model_endpoint(
        model_endpoint_id=model_endpoint_record.id,
    )

    # The model endpoint service should no longer return the record.
    deleted_model_endpoint_record = await fake_live_model_endpoint_service.get_model_endpoint(
        model_endpoint_id=model_endpoint_record.id
    )
    assert deleted_model_endpoint_record is None


@pytest.mark.asyncio
async def test_create_delete_model_endpoint_infra_not_deleted_raises_endpoint_delete_failed_exception(
    fake_live_model_endpoint_service: LiveModelEndpointService,
    model_endpoint_1: ModelEndpoint,
):
    # Create the model endpoint.
    model_endpoint_record = await _create_model_endpoint_helper(
        model_endpoint=model_endpoint_1, service=fake_live_model_endpoint_service
    )

    # Mock a failed delete operation.
    fake_live_model_endpoint_service.model_endpoint_infra_gateway.__setattr__(
        "delete_model_endpoint_infra",
        AsyncMock(return_value=False),
    )

    # Deleting the model endpoint before it's ready should raise an exception.
    with pytest.raises(EndpointDeleteFailedException):
        await fake_live_model_endpoint_service.delete_model_endpoint(
            model_endpoint_id=model_endpoint_record.id,
        )


@pytest.mark.asyncio
async def test_delete_model_endpoint_raises_not_found(
    fake_live_model_endpoint_service: LiveModelEndpointService,
):
    with pytest.raises(ObjectNotFoundException):
        await fake_live_model_endpoint_service.delete_model_endpoint(
            model_endpoint_id="invalid_model_endpoint_id",
        )
