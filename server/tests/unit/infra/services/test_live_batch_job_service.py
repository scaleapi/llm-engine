import pytest

from llm_engine_server.common.dtos.batch_jobs import CreateBatchJobResourceRequests
from llm_engine_server.domain.entities import BatchJobSerializationFormat, GpuType, ModelBundle
from llm_engine_server.domain.exceptions import EndpointResourceInvalidRequestException
from llm_engine_server.infra.services import LiveBatchJobService
from llm_engine_server.infra.services.live_batch_job_service import (
    DEFAULT_ENDPOINT_CPUS_BATCH_JOB,
    DEFAULT_ENDPOINT_GPU_TYPE_BATCH_JOB,
    DEFAULT_ENDPOINT_GPUS_BATCH_JOB,
    DEFAULT_ENDPOINT_MAX_WORKERS_BATCH_JOB,
    DEFAULT_ENDPOINT_MEMORY_BATCH_JOB,
    DEFAULT_ENDPOINT_PER_WORKER_BATCH_JOB,
)


@pytest.fixture
def create_batch_job_resource_requests_1() -> CreateBatchJobResourceRequests:
    return CreateBatchJobResourceRequests(
        cpus=3,
        memory="12Gi",
        gpus=1,
        gpu_type=None,
        storage="100Gi",
        max_workers=50,
        per_worker=40,
    )


@pytest.fixture
def create_batch_job_resource_requests_2() -> CreateBatchJobResourceRequests:
    return CreateBatchJobResourceRequests(
        cpus=3,
        memory="12Gi",
        gpus=0,
        gpu_type=None,
        storage="100Gi",
        max_workers=50,
        per_worker=40,
    )


@pytest.fixture
def create_batch_job_resource_requests_3() -> CreateBatchJobResourceRequests:
    return CreateBatchJobResourceRequests(
        cpus=3,
        memory="12Gi",
        gpus=0,
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage="100Gi",
        max_workers=50,
        per_worker=40,
    )


@pytest.fixture
def create_batch_job_resource_requests_4() -> CreateBatchJobResourceRequests:
    return CreateBatchJobResourceRequests(
        cpus=None,
        memory=None,
        gpus=None,
        gpu_type=None,
        storage="100Gi",
        max_workers=None,
        per_worker=None,
    )


@pytest.mark.asyncio
async def test_create_batch_job_populate_default_gpu_type_when_gpus_gt_0(
    create_batch_job_resource_requests_1: CreateBatchJobResourceRequests,
    fake_live_batch_job_service: LiveBatchJobService,
    model_bundle_1: ModelBundle,
):
    batch_job_id = await fake_live_batch_job_service.create_batch_job(
        created_by="test-user",
        owner="test-user",
        model_bundle_id=model_bundle_1.id,
        input_path="test-input-path",
        serialization_format=BatchJobSerializationFormat.JSON,
        labels={},
        resource_requests=create_batch_job_resource_requests_1,
        aws_role="test-aws-role",
        results_s3_bucket="test-results-s3-bucket",
        timeout_seconds=1800.0,
    )
    batch_job_record = (
        await fake_live_batch_job_service.batch_job_record_repository.get_batch_job_record(
            batch_job_id
        )
    )
    assert batch_job_record is not None
    assert batch_job_record.model_endpoint_id is not None
    model_endpoint = await fake_live_batch_job_service.model_endpoint_service.get_model_endpoint(
        batch_job_record.model_endpoint_id
    )
    assert model_endpoint is not None
    assert model_endpoint.infra_state.resource_state.gpu_type == DEFAULT_ENDPOINT_GPU_TYPE_BATCH_JOB


@pytest.mark.asyncio
async def test_create_batch_job_do_not_populate_default_gpu_type_when_gpus_is_0(
    create_batch_job_resource_requests_2: CreateBatchJobResourceRequests,
    fake_live_batch_job_service: LiveBatchJobService,
    model_bundle_1: ModelBundle,
):
    batch_job_id = await fake_live_batch_job_service.create_batch_job(
        created_by="test-user",
        owner="test-user",
        model_bundle_id=model_bundle_1.id,
        input_path="test-input-path",
        serialization_format=BatchJobSerializationFormat.JSON,
        labels={},
        resource_requests=create_batch_job_resource_requests_2,
        aws_role="test-aws-role",
        results_s3_bucket="test-results-s3-bucket",
        timeout_seconds=1800.0,
    )
    batch_job_record = (
        await fake_live_batch_job_service.batch_job_record_repository.get_batch_job_record(
            batch_job_id
        )
    )
    assert batch_job_record is not None
    assert batch_job_record.model_endpoint_id is not None
    model_endpoint = await fake_live_batch_job_service.model_endpoint_service.get_model_endpoint(
        batch_job_record.model_endpoint_id
    )
    assert model_endpoint is not None
    assert model_endpoint.infra_state.resource_state.gpu_type is None


@pytest.mark.asyncio
async def test_create_batch_job_raise_value_error_when_requsting_gpu_type_with_0_gpus(
    create_batch_job_resource_requests_3: CreateBatchJobResourceRequests,
    fake_live_batch_job_service: LiveBatchJobService,
    model_bundle_1: ModelBundle,
):
    with pytest.raises(EndpointResourceInvalidRequestException):
        await fake_live_batch_job_service.create_batch_job(
            created_by="test-user",
            owner="test-user",
            model_bundle_id=model_bundle_1.id,
            input_path="test-input-path",
            serialization_format=BatchJobSerializationFormat.JSON,
            labels={},
            resource_requests=create_batch_job_resource_requests_3,
            aws_role="test-aws-role",
            results_s3_bucket="test-results-s3-bucket",
            timeout_seconds=1800.0,
        )


@pytest.mark.asyncio
async def test_create_batch_job_populate_default_values(
    create_batch_job_resource_requests_4: CreateBatchJobResourceRequests,
    fake_live_batch_job_service: LiveBatchJobService,
    model_bundle_1: ModelBundle,
):
    batch_job_id = await fake_live_batch_job_service.create_batch_job(
        created_by="test-user",
        owner="test-user",
        model_bundle_id=model_bundle_1.id,
        input_path="test-input-path",
        serialization_format=BatchJobSerializationFormat.JSON,
        labels={},
        resource_requests=create_batch_job_resource_requests_4,
        aws_role="test-aws-role",
        results_s3_bucket="test-results-s3-bucket",
        timeout_seconds=1800.0,
    )
    batch_job_record = (
        await fake_live_batch_job_service.batch_job_record_repository.get_batch_job_record(
            batch_job_id
        )
    )
    assert batch_job_record is not None
    assert batch_job_record.model_endpoint_id is not None
    model_endpoint = await fake_live_batch_job_service.model_endpoint_service.get_model_endpoint(
        batch_job_record.model_endpoint_id
    )
    assert model_endpoint is not None
    assert int(model_endpoint.infra_state.resource_state.cpus) == DEFAULT_ENDPOINT_CPUS_BATCH_JOB
    assert model_endpoint.infra_state.resource_state.memory == DEFAULT_ENDPOINT_MEMORY_BATCH_JOB
    assert model_endpoint.infra_state.resource_state.gpus == DEFAULT_ENDPOINT_GPUS_BATCH_JOB
    assert model_endpoint.infra_state.resource_state.gpu_type == DEFAULT_ENDPOINT_GPU_TYPE_BATCH_JOB
    assert (
        model_endpoint.infra_state.deployment_state.max_workers
        == DEFAULT_ENDPOINT_MAX_WORKERS_BATCH_JOB
    )
    assert (
        model_endpoint.infra_state.deployment_state.per_worker
        == DEFAULT_ENDPOINT_PER_WORKER_BATCH_JOB
    )
