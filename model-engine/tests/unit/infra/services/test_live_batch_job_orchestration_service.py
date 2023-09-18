import json
from datetime import timedelta
from typing import Any
from unittest.mock import patch

import pytest
from model_engine_server.common.constants import DEFAULT_CELERY_TASK_NAME, LIRA_CELERY_TASK_NAME
from model_engine_server.common.dtos.tasks import GetAsyncTaskV1Response, ResponseSchema, TaskStatus
from model_engine_server.domain.entities import (
    BatchJob,
    BatchJobSerializationFormat,
    BatchJobStatus,
    ModelBundle,
    ModelEndpoint,
    ModelEndpointStatus,
)
from model_engine_server.domain.exceptions import ObjectNotFoundException
from model_engine_server.infra.gateways import LiveBatchJobProgressGateway
from model_engine_server.infra.services import (
    LiveBatchJobOrchestrationService,
    LiveModelEndpointService,
)
from model_engine_server.infra.services.live_batch_job_orchestration_service import (
    BatchEndpointInferencePredictionResponse,
    BatchEndpointInProgressTask,
)


@pytest.fixture
def live_batch_job_orchestration_service(
    fake_live_model_endpoint_service: LiveModelEndpointService,
    fake_batch_job_record_repository,
    fake_async_model_endpoint_inference_gateway,
    fake_model_bundle_repository,
    fake_filesystem_gateway,
    model_bundle_1: ModelBundle,  # already inserted into fake_live_model_endpoint_service fixture.
    model_bundle_4: ModelBundle,  # not already added so we add it here
    model_endpoint_1: ModelEndpoint,
    model_endpoint_runnable: ModelEndpoint,
    batch_job_1: BatchJob,
    batch_job_from_runnable: BatchJob,
) -> LiveBatchJobOrchestrationService:
    fake_model_bundle_repository.add_model_bundle(model_bundle_4)
    fake_batch_job_record_repository.add_batch_job_record(batch_job_1.record)
    fake_batch_job_record_repository.add_batch_job_record(batch_job_from_runnable.record)
    repo: Any = fake_live_model_endpoint_service.model_endpoint_record_repository
    repo.add_model_endpoint_record(model_endpoint_1.record)
    repo.add_model_endpoint_record(model_endpoint_runnable.record)
    gateway: Any = fake_live_model_endpoint_service.model_endpoint_infra_gateway
    assert model_endpoint_1.infra_state is not None
    assert model_endpoint_runnable.infra_state is not None
    gateway.db[model_endpoint_1.infra_state.deployment_name] = model_endpoint_1.infra_state
    gateway.db[
        model_endpoint_runnable.infra_state.deployment_name
    ] = model_endpoint_runnable.infra_state
    return LiveBatchJobOrchestrationService(
        model_endpoint_service=fake_live_model_endpoint_service,
        batch_job_record_repository=fake_batch_job_record_repository,
        batch_job_progress_gateway=LiveBatchJobProgressGateway(
            filesystem_gateway=fake_filesystem_gateway
        ),
        async_model_endpoint_inference_gateway=fake_async_model_endpoint_inference_gateway,
        filesystem_gateway=fake_filesystem_gateway,
    )


@pytest.fixture
def batch_job_input() -> str:
    data = [("url",), ("url1",), ("url2",)]
    return "\n".join([",".join([str(elt) for elt in datum]) for datum in data])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "serialization_format",
    [BatchJobSerializationFormat.JSON, BatchJobSerializationFormat.PICKLE],
)
async def test_run_batch_job_json_success(
    live_batch_job_orchestration_service: LiveBatchJobOrchestrationService,
    batch_job_1: BatchJob,
    batch_job_input: str,
    serialization_format: BatchJobSerializationFormat,
):
    gateway: Any = live_batch_job_orchestration_service.filesystem_gateway
    gateway.read_data = batch_job_input
    model_endpoint_id = batch_job_1.record.model_endpoint_id

    assert (
        not batch_job_1.record.model_bundle.is_runnable()
    ), "Test is broken, the batch job's bundle should be not runnable"

    await live_batch_job_orchestration_service.run_batch_job(
        job_id=batch_job_1.record.id,
        owner=batch_job_1.record.owner,
        input_path="test_input",
        serialization_format=serialization_format,
        timeout=timedelta(hours=12),
    )

    batch_job = (
        await live_batch_job_orchestration_service.batch_job_record_repository.get_batch_job_record(
            batch_job_id=batch_job_1.record.id,
        )
    )
    assert batch_job is not None
    assert batch_job.status == BatchJobStatus.SUCCESS

    # model_bundle_1 is cloudpickle type, assert that we use the default celery task name
    fake_async_model_endpoint_inference_gateway = (
        live_batch_job_orchestration_service.async_model_endpoint_inference_gateway
    )
    assert (
        fake_async_model_endpoint_inference_gateway.get_last_request().task_name
        == DEFAULT_CELERY_TASK_NAME
    )

    # Assert that the model endpoint was deleted.
    assert model_endpoint_id is not None
    model_endpoint = (
        await live_batch_job_orchestration_service.model_endpoint_service.get_model_endpoint(
            model_endpoint_id=model_endpoint_id,
        )
    )
    assert model_endpoint is None


@pytest.mark.asyncio
async def test_run_batch_job_runnable_uses_lira_request(
    live_batch_job_orchestration_service: LiveBatchJobOrchestrationService,
    batch_job_from_runnable: BatchJob,
    batch_job_input: str,
):
    gateway: Any = live_batch_job_orchestration_service.filesystem_gateway
    gateway.read_data = batch_job_input
    model_endpoint_id = batch_job_from_runnable.record.model_endpoint_id

    assert (
        batch_job_from_runnable.record.model_bundle.is_runnable()
    ), "Test is broken, the batch job's bundle should be runnable"

    await live_batch_job_orchestration_service.run_batch_job(
        job_id=batch_job_from_runnable.record.id,
        owner=batch_job_from_runnable.record.owner,
        input_path="test_input",
        serialization_format=BatchJobSerializationFormat.JSON,
        timeout=timedelta(hours=12),
    )

    batch_job = (
        await live_batch_job_orchestration_service.batch_job_record_repository.get_batch_job_record(
            batch_job_id=batch_job_from_runnable.record.id,
        )
    )
    assert batch_job is not None
    assert batch_job.status == BatchJobStatus.SUCCESS

    # batch_job_from_runnable's bundle is runnable type, assert we use the lira celery task name
    fake_async_model_endpoint_inference_gateway = (
        live_batch_job_orchestration_service.async_model_endpoint_inference_gateway
    )
    assert (
        fake_async_model_endpoint_inference_gateway.get_last_request().task_name
        == LIRA_CELERY_TASK_NAME
    )

    # Assert that the model endpoint was deleted.
    assert model_endpoint_id is not None
    model_endpoint = (
        await live_batch_job_orchestration_service.model_endpoint_service.get_model_endpoint(
            model_endpoint_id=model_endpoint_id,
        )
    )
    assert model_endpoint is None


@pytest.mark.asyncio
async def test_run_batch_job_failure_sets_status(
    live_batch_job_orchestration_service: LiveBatchJobOrchestrationService,
    batch_job_1: BatchJob,
):
    gateway: Any = live_batch_job_orchestration_service.filesystem_gateway
    gateway.read_data = b"invalid json"
    await live_batch_job_orchestration_service.run_batch_job(
        job_id=batch_job_1.record.id,
        owner=batch_job_1.record.owner,
        input_path="test_input",
        serialization_format=BatchJobSerializationFormat.JSON,
        timeout=timedelta(hours=12),
    )

    batch_job = (
        await live_batch_job_orchestration_service.batch_job_record_repository.get_batch_job_record(
            batch_job_id=batch_job_1.record.id,
        )
    )
    assert batch_job is not None
    assert batch_job.status == BatchJobStatus.FAILURE


@pytest.mark.asyncio
async def test_run_batch_job_not_found_raises_not_found(
    live_batch_job_orchestration_service: LiveBatchJobOrchestrationService,
):
    with pytest.raises(ObjectNotFoundException):
        await live_batch_job_orchestration_service.run_batch_job(
            job_id="invalid_job_id",
            owner="invalid_owner",
            input_path="test_input",
            serialization_format=BatchJobSerializationFormat.JSON,
            timeout=timedelta(hours=12),
        )


@pytest.mark.asyncio
async def test_run_batch_job_endpoint_not_found_raises_not_found(
    live_batch_job_orchestration_service: LiveBatchJobOrchestrationService,
    batch_job_1: BatchJob,
):
    batch_job_1.record.model_endpoint_id = None
    with pytest.raises(ObjectNotFoundException):
        await live_batch_job_orchestration_service.run_batch_job(
            job_id=batch_job_1.record.id,
            owner=batch_job_1.record.owner,
            input_path="test_input",
            serialization_format=BatchJobSerializationFormat.JSON,
            timeout=timedelta(hours=12),
        )


@pytest.mark.asyncio
async def test_run_batch_job_wait_for_endpoint(
    live_batch_job_orchestration_service: LiveBatchJobOrchestrationService,
    batch_job_1: BatchJob,
    model_endpoint_1: ModelEndpoint,
):
    model_endpoint_1.record.status = ModelEndpointStatus.UPDATE_PENDING
    with patch(
        "model_engine_server.infra.services.live_batch_job_orchestration_service.asyncio.sleep"
    ) as mock_sleep:

        def set_record_ready(*args, **kwargs):
            model_endpoint_1.record.status = ModelEndpointStatus.READY

        mock_sleep.side_effect = set_record_ready
        await live_batch_job_orchestration_service.run_batch_job(
            job_id=batch_job_1.record.id,
            owner=batch_job_1.record.owner,
            input_path="test_input",
            serialization_format=BatchJobSerializationFormat.JSON,
            timeout=timedelta(hours=12),
        )
        mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_run_batch_job_wait_for_endpoint_raises_object_not_found(
    live_batch_job_orchestration_service: LiveBatchJobOrchestrationService,
    batch_job_1: BatchJob,
    model_endpoint_1: ModelEndpoint,
):
    model_endpoint_1.record.status = ModelEndpointStatus.UPDATE_FAILED
    with pytest.raises(ObjectNotFoundException):
        await live_batch_job_orchestration_service.run_batch_job(
            job_id=batch_job_1.record.id,
            owner=batch_job_1.record.owner,
            input_path="test_input",
            serialization_format=BatchJobSerializationFormat.JSON,
            timeout=timedelta(hours=12),
        )


def test_in_progress_task_serialization_deserialization():
    task = BatchEndpointInProgressTask("task_id", "ref_id")
    assert task == BatchEndpointInProgressTask.deserialize(task.serialize())


def test_prediction_response_json_encoding():
    pred_response_inner = GetAsyncTaskV1Response(
        task_id="task_id",
        status=TaskStatus.SUCCESS,
        result=ResponseSchema.parse_obj({"some": "custom", "returned": "value"}),
        traceback="traceback",
    )
    pred_response = BatchEndpointInferencePredictionResponse(
        response=pred_response_inner, reference_id="refid"
    )
    assert json.loads(pred_response.json()) == dict(
        task_id="task_id",
        status="SUCCESS",
        result=dict(some="custom", returned="value"),
        traceback="traceback",
        id="refid",
    )
