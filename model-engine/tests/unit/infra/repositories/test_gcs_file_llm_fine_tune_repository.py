import io
import json
from unittest import mock

import pytest
from model_engine_server.domain.entities.llm_fine_tune_entity import LLMFineTuneTemplate
from model_engine_server.infra.repositories.gcs_file_llm_fine_tune_repository import (
    GCSFileLLMFineTuneRepository,
)

FAKE_FILE_PATH = "gs://fake-bucket/fine-tune-templates.json"
MODULE = "model_engine_server.infra.repositories.gcs_file_llm_fine_tune_repository"

SAMPLE_TEMPLATE = {
    "docker_image_batch_job_bundle_id": "bundle-123",
    "launch_endpoint_config": {"cpus": 4},
    "default_hparams": {"lr": 0.001},
    "required_params": ["training_file"],
}


@pytest.fixture
def repository():
    return GCSFileLLMFineTuneRepository(file_path=FAKE_FILE_PATH)


def _make_mock_open(data: dict):
    """Create a mock for _open that returns a file-like object with the given JSON data."""

    def mock_open_fn(uri, mode="rt", **kwargs):
        if "r" in mode:
            return io.StringIO(json.dumps(data))
        else:
            return io.StringIO()

    return mock_open_fn


@pytest.mark.asyncio
@mock.patch(f"{MODULE}.get_gcs_sync_client")
async def test_get_job_template_found(mock_client, repository):
    data = {"model1-lora": SAMPLE_TEMPLATE}
    repository._open = mock.Mock(side_effect=_make_mock_open(data))

    result = await repository.get_job_template_for_model("model1", "lora")
    assert result is not None
    assert isinstance(result, LLMFineTuneTemplate)
    assert result.docker_image_batch_job_bundle_id == "bundle-123"
    assert result.required_params == ["training_file"]


@pytest.mark.asyncio
@mock.patch(f"{MODULE}.get_gcs_sync_client")
async def test_get_job_template_not_found(mock_client, repository):
    data = {"other-key": SAMPLE_TEMPLATE}
    repository._open = mock.Mock(side_effect=_make_mock_open(data))

    result = await repository.get_job_template_for_model("model1", "lora")
    assert result is None


@pytest.mark.asyncio
@mock.patch(f"{MODULE}.get_gcs_sync_client")
async def test_write_job_template(mock_client, repository):
    existing_data = {"existing-key": SAMPLE_TEMPLATE}
    written = io.StringIO()

    call_count = 0

    def mock_open_fn(uri, mode="rt", **kwargs):
        nonlocal call_count
        call_count += 1
        if "r" in mode:
            return io.StringIO(json.dumps(existing_data))
        else:
            return written

    repository._open = mock.Mock(side_effect=mock_open_fn)

    template = LLMFineTuneTemplate(**SAMPLE_TEMPLATE)
    await repository.write_job_template_for_model("model2", "qlora", template)

    written.seek(0)
    result = json.loads(written.getvalue())
    assert "model2-qlora" in result
    assert "existing-key" in result


@pytest.mark.asyncio
@mock.patch(f"{MODULE}.get_gcs_sync_client")
async def test_initialize_data(mock_client, repository):
    written = io.StringIO()
    repository._open = mock.Mock(return_value=written)

    await repository.initialize_data()

    written.seek(0)
    assert json.loads(written.getvalue()) == {}
