from spellbook_serve.domain.entities import BatchJobProgress
from spellbook_serve.infra.gateways import LiveBatchJobProgressGateway


def test_get_progress_empty(test_api_key: str, fake_filesystem_gateway):
    fake_filesystem_gateway.read_data = ""
    live_batch_job_progress_gateway = LiveBatchJobProgressGateway(
        filesystem_gateway=fake_filesystem_gateway
    )
    initial_result = live_batch_job_progress_gateway.get_progress(
        owner=test_api_key, batch_job_id="job_id"
    )
    assert initial_result == BatchJobProgress()


def test_get_progress(test_api_key: str, fake_filesystem_gateway):
    fake_filesystem_gateway.read_data = '{"num_tasks_pending": 4, "num_tasks_completed": 5}'
    live_batch_job_progress_gateway = LiveBatchJobProgressGateway(
        filesystem_gateway=fake_filesystem_gateway
    )
    new_result = live_batch_job_progress_gateway.get_progress(
        owner=test_api_key, batch_job_id="job_id"
    )
    assert new_result == BatchJobProgress(num_tasks_pending=4, num_tasks_completed=5)


def test_update_progress(test_api_key: str, fake_filesystem_gateway):
    live_batch_job_progress_gateway = LiveBatchJobProgressGateway(
        filesystem_gateway=fake_filesystem_gateway
    )
    live_batch_job_progress_gateway.update_progress(
        owner=test_api_key,
        batch_job_id="job_id",
        progress=BatchJobProgress(num_tasks_pending=4, num_tasks_completed=5),
    )
    handle = fake_filesystem_gateway.mock_open()
    handle.write.assert_called_once_with('{"num_tasks_pending": 4, "num_tasks_completed": 5}')
