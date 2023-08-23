import pytest
from model_engine_server.infra.services.model_endpoint_cache_service import (
    ModelEndpointCacheWriteService,
)


@pytest.mark.asyncio
async def test_model_endpoint_write_success(
    fake_model_endpoint_cache_repository,
    fake_resource_gateway,
    fake_image_cache_service,
    model_endpoint_1,
    model_endpoint_2,
):
    # Model endpoint 1 exists, 2 doesn't
    fake_resource_gateway.add_resource(
        endpoint_id=model_endpoint_1.record.id, infra_state=model_endpoint_1.infra_state
    )

    cache_write_service = ModelEndpointCacheWriteService(
        fake_model_endpoint_cache_repository, fake_resource_gateway, fake_image_cache_service
    )
    await cache_write_service.execute(42)
    infra_state = await fake_model_endpoint_cache_repository.read_endpoint_info(
        endpoint_id=model_endpoint_1.record.id,
        deployment_name=model_endpoint_1.infra_state.deployment_name,
    )
    assert infra_state == model_endpoint_1.infra_state
    infra_state = await fake_model_endpoint_cache_repository.read_endpoint_info(
        endpoint_id=model_endpoint_2.record.id,
        deployment_name=model_endpoint_2.infra_state.deployment_name,
    )
    assert infra_state is None

    fake_resource_gateway.add_resource(
        endpoint_id=model_endpoint_2.record.id, infra_state=model_endpoint_2.infra_state
    )
    # Model endpoint 2 hasn't updated yet
    infra_state = await fake_model_endpoint_cache_repository.read_endpoint_info(
        endpoint_id=model_endpoint_2.record.id,
        deployment_name=model_endpoint_2.infra_state.deployment_name,
    )
    assert infra_state is None

    # Update model endpoint 2 in cache
    await cache_write_service.execute(43)
    infra_state = await fake_model_endpoint_cache_repository.read_endpoint_info(
        endpoint_id=model_endpoint_2.record.id,
        deployment_name=model_endpoint_2.infra_state.deployment_name,
    )
    assert infra_state == model_endpoint_2.infra_state

    # Expire model endpoint 1
    infra_state = await fake_model_endpoint_cache_repository.read_endpoint_info(
        endpoint_id=model_endpoint_1.record.id,
        deployment_name=model_endpoint_1.infra_state.deployment_name,
    )
    assert infra_state == model_endpoint_1.infra_state
    fake_model_endpoint_cache_repository.force_expire_key(
        endpoint_id=model_endpoint_1.record.id,
    )
    infra_state = await fake_model_endpoint_cache_repository.read_endpoint_info(
        endpoint_id=model_endpoint_1.record.id,
        deployment_name=model_endpoint_1.infra_state.deployment_name,
    )
    assert infra_state is None
