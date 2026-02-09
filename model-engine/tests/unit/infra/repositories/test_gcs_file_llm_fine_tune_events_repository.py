import io
from unittest import mock

import pytest
from model_engine_server.domain.exceptions import ObjectNotFoundException
from model_engine_server.infra.repositories.gcs_file_llm_fine_tune_events_repository import (
    GCSFileLLMFineTuneEventsRepository,
)

MODULE = "model_engine_server.infra.repositories.gcs_file_llm_fine_tune_events_repository"


@pytest.fixture
def repository():
    return GCSFileLLMFineTuneEventsRepository()


@pytest.mark.asyncio
@mock.patch(f"{MODULE}.get_gcs_sync_client")
async def test_get_fine_tune_events(mock_client, repository):
    jsonl_content = (
        '{"timestamp": 1700000000.0, "message": "Starting training", "level": "info"}\n'
        '{"timestamp": 1700000060.0, "message": "Epoch 1 complete", "level": "info"}\n'
    )
    repository._open = mock.Mock(return_value=io.StringIO(jsonl_content))

    events = await repository.get_fine_tune_events("user1", "my-model")
    assert len(events) == 2
    assert events[0].message == "Starting training"
    assert events[0].timestamp == 1700000000.0
    assert events[1].message == "Epoch 1 complete"


@pytest.mark.asyncio
@mock.patch(f"{MODULE}.get_gcs_sync_client")
async def test_get_fine_tune_events_malformed_json(mock_client, repository):
    jsonl_content = "this is not json\n"
    repository._open = mock.Mock(return_value=io.StringIO(jsonl_content))

    events = await repository.get_fine_tune_events("user1", "my-model")
    assert len(events) == 1
    assert events[0].message == "this is not json\n"
    assert events[0].level == "info"


@pytest.mark.asyncio
@mock.patch(f"{MODULE}.get_gcs_sync_client")
async def test_get_fine_tune_events_not_found(mock_client, repository):
    repository._open = mock.Mock(side_effect=FileNotFoundError("not found"))

    with pytest.raises(ObjectNotFoundException):
        await repository.get_fine_tune_events("user1", "nonexistent-model")


@pytest.mark.asyncio
@mock.patch(f"{MODULE}.get_gcs_sync_client")
async def test_initialize_events(mock_client, repository):
    repository._open = mock.Mock(return_value=io.StringIO())

    await repository.initialize_events("user1", "my-model")

    repository._open.assert_called_once()
    call_args = repository._open.call_args
    assert call_args[1].get("mode", call_args[0][1] if len(call_args[0]) > 1 else None) == "w"
