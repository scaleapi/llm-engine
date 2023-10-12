from unittest.mock import AsyncMock, patch

import pytest
from model_engine_server.domain.entities import BatchJobStatus
from model_engine_server.infra.gateways.live_docker_image_batch_job_gateway import (
    K8sEnvDict,
    LiveDockerImageBatchJobGateway,
    _add_list_values,
    _check_batch_job_id_valid,
    _get_job_id,
)
from tests.unit.infra.gateways.k8s_fake_objects import (
    FakeK8sV1Job,
    FakeK8sV1JobList,
    FakeK8sV1JobStatus,
    FakeK8sV1ObjectMeta,
    FakeK8sV1Pod,
    FakeK8sV1PodList,
    FakeK8sV1PodStatus,
)

MODULE_PATH = "model_engine_server.infra.gateways.live_docker_image_batch_job_gateway"


@pytest.fixture
def mock_core_client():
    mock_client = AsyncMock()
    with patch(
        f"{MODULE_PATH}.get_kubernetes_core_client",
        return_value=mock_client,
    ):
        yield mock_client


@pytest.fixture
def mock_batch_client():
    mock_client = AsyncMock()
    with patch(
        f"{MODULE_PATH}.get_kubernetes_batch_client",
        return_value=mock_client,
    ):
        yield mock_client


@pytest.fixture
def docker_image_batch_job_gateway():
    gateway = LiveDockerImageBatchJobGateway()
    return gateway


@pytest.mark.parametrize(
    "active, succeeded, failed, pod_phase, pod_exists, expected_status",
    [
        [1, 0, 0, "Running", True, BatchJobStatus.RUNNING],
        [0, 1, 0, "Succeeded", True, BatchJobStatus.SUCCESS],
        [0, 0, 1, "Failed", True, BatchJobStatus.FAILURE],
        [1, 0, 0, "Pending", True, BatchJobStatus.PENDING],
        [0, 0, 0, "Pending", False, BatchJobStatus.PENDING],
    ],
)
@pytest.mark.asyncio
async def test_get_docker_image_batch_job_phase(
    active,
    succeeded,
    failed,
    pod_phase,
    pod_exists,
    expected_status,
    docker_image_batch_job_gateway,
    mock_core_client,
    mock_batch_client,
):
    if pod_exists:
        pod_items = [
            FakeK8sV1Pod(
                metadata=FakeK8sV1ObjectMeta(
                    labels={
                        "job-name": "job-name",
                        "owner": "owner",
                        "created_by": "created_by",
                        "trigger_id": "trigger_id",
                        "launch_job_id": "launch_job_id",
                    }
                ),
                status=FakeK8sV1PodStatus(
                    phase=pod_phase,
                ),
            )
        ]
    else:
        pod_items = []

    mock_core_client.list_namespaced_pod.return_value = FakeK8sV1PodList(items=pod_items)
    mock_batch_client.list_namespaced_job.return_value = FakeK8sV1JobList(
        items=[
            FakeK8sV1Job(
                metadata=FakeK8sV1ObjectMeta(
                    name="job-name",
                    labels={
                        "owner": "owner",
                        "created_by": "created_by",
                        "trigger_id": "trigger_id",
                        "launch_job_id": "launch_job_id",
                    },
                ),
                status=FakeK8sV1JobStatus(
                    active=active,
                    succeeded=succeeded,
                    failed=failed,
                ),
            )
        ]
    )

    job = await docker_image_batch_job_gateway.get_docker_image_batch_job("launch_job_id")
    assert job is not None
    assert job.status == expected_status


# Small function functionality tests
def test_valid_job_ids_are_valid():
    for _ in range(20):
        # _get_job_id() is nondeterministic
        job_id = _get_job_id()
        assert _check_batch_job_id_valid(job_id), f"job_id {job_id} apparently isn't valid"


def test_invalid_job_ids_are_invalid():
    assert not _check_batch_job_id_valid("spaces fail")
    assert not _check_batch_job_id_valid("punctuation'")
    assert not _check_batch_job_id_valid(".")


# test the adding list values
def test_add_list_values():
    default_values = [
        K8sEnvDict(name="default1", value="val1"),
        K8sEnvDict(name="default2", value="val2"),
        K8sEnvDict(name="default3", value="val3"),
    ]
    override_values = [
        K8sEnvDict(name="default1", value="override0"),
        K8sEnvDict(name="override1", value="override1"),
        K8sEnvDict(name="override2", value="override2"),
    ]
    expected_values = [
        K8sEnvDict(name="default1", value="val1"),
        K8sEnvDict(name="default2", value="val2"),
        K8sEnvDict(name="default3", value="val3"),
        K8sEnvDict(name="override1", value="override1"),
        K8sEnvDict(name="override2", value="override2"),
    ]

    actual_values = _add_list_values(default_values, override_values)
    actual_values.sort(key=lambda x: x["name"])
    assert expected_values == actual_values
