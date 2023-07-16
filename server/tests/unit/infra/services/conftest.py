import pytest

from spellbook_serve.domain.entities import ModelBundle, ModelEndpoint
from spellbook_serve.infra.gateways import (
    LiveBatchJobProgressGateway,
    LiveModelEndpointsSchemaGateway,
)
from spellbook_serve.infra.services import LiveBatchJobService, LiveModelEndpointService


@pytest.fixture
def fake_live_model_endpoint_service(
    fake_model_endpoint_record_repository,
    fake_model_endpoint_infra_gateway,
    fake_model_bundle_repository,
    fake_model_endpoint_cache_repository,
    fake_async_model_endpoint_inference_gateway,
    fake_streaming_model_endpoint_inference_gateway,
    fake_sync_model_endpoint_inference_gateway,
    fake_filesystem_gateway,
    model_bundle_1: ModelBundle,
    model_bundle_2: ModelBundle,
    model_endpoint_1: ModelEndpoint,
) -> LiveModelEndpointService:
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_bundle_repository.add_model_bundle(model_bundle_2)
    fake_model_endpoint_record_repository.model_bundle_repository = fake_model_bundle_repository
    fake_model_endpoint_infra_gateway.model_endpoint_record_repository = (
        fake_model_endpoint_record_repository
    )
    model_endpoints_schema_gateway = LiveModelEndpointsSchemaGateway(
        filesystem_gateway=fake_filesystem_gateway,
    )
    service = LiveModelEndpointService(
        model_endpoint_record_repository=fake_model_endpoint_record_repository,
        model_endpoint_infra_gateway=fake_model_endpoint_infra_gateway,
        model_endpoint_cache_repository=fake_model_endpoint_cache_repository,
        async_model_endpoint_inference_gateway=fake_async_model_endpoint_inference_gateway,
        streaming_model_endpoint_inference_gateway=fake_streaming_model_endpoint_inference_gateway,
        sync_model_endpoint_inference_gateway=fake_sync_model_endpoint_inference_gateway,
        model_endpoints_schema_gateway=model_endpoints_schema_gateway,
    )
    return service


@pytest.fixture
def fake_live_batch_job_service(
    fake_batch_job_record_repository,
    fake_batch_job_orchestration_gateway,
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    fake_filesystem_gateway,
    model_bundle_1: ModelBundle,
) -> LiveBatchJobService:
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_batch_job_record_repository.model_bundle_repository = fake_model_bundle_repository
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    service = LiveBatchJobService(
        batch_job_record_repository=fake_batch_job_record_repository,
        model_endpoint_service=fake_model_endpoint_service,
        batch_job_orchestration_gateway=fake_batch_job_orchestration_gateway,
        batch_job_progress_gateway=LiveBatchJobProgressGateway(
            filesystem_gateway=fake_filesystem_gateway
        ),
    )
    return service
