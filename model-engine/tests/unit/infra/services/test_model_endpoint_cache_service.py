import pytest
from model_engine_server.infra.services.model_endpoint_cache_service import (
    ModelEndpointCacheWriteService,
)
from tests.unit.conftest import FakeModelEndpointCacheRepository


@pytest.mark.asyncio
async def test_model_endpoint_write_success(
    fake_model_endpoint_cache_repository,
    fake_resource_gateway,
    fake_image_cache_service,
    fake_monitoring_metrics_gateway,
    model_endpoint_1,
    model_endpoint_2,
):
    # Model endpoint 1 exists, 2 doesn't
    fake_resource_gateway.add_resource(
        endpoint_id=model_endpoint_1.record.id, infra_state=model_endpoint_1.infra_state
    )

    cache_write_service = ModelEndpointCacheWriteService(
        fake_model_endpoint_cache_repository,
        fake_resource_gateway,
        fake_image_cache_service,
        fake_monitoring_metrics_gateway,
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

    # Happy path emits no write-failure metric.
    assert fake_monitoring_metrics_gateway.cache_write_failure == 0


class _RaisingCacheRepository(FakeModelEndpointCacheRepository):
    """Simulates Redis being unwritable (e.g. bad auth / network partition)."""

    async def write_endpoint_info(self, endpoint_id, endpoint_info, ttl_seconds):
        raise ConnectionError("Error connecting to Redis")


@pytest.mark.asyncio
async def test_model_endpoint_write_failure_emits_metric_and_reraises(
    fake_resource_gateway,
    fake_image_cache_service,
    fake_monitoring_metrics_gateway,
    model_endpoint_1,
):
    fake_resource_gateway.add_resource(
        endpoint_id=model_endpoint_1.record.id, infra_state=model_endpoint_1.infra_state
    )

    cache_write_service = ModelEndpointCacheWriteService(
        _RaisingCacheRepository(),
        fake_resource_gateway,
        fake_image_cache_service,
        fake_monitoring_metrics_gateway,
    )

    with pytest.raises(ConnectionError):
        await cache_write_service.execute(42)

    assert fake_monitoring_metrics_gateway.cache_write_failure == 1
