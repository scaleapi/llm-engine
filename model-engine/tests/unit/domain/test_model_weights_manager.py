"""Unit tests for ModelWeightsManager."""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from model_engine_server.domain.gateways.llm_artifact_gateway import LLMArtifactGateway
from model_engine_server.domain.use_cases.model_weights_manager import (
    HF_IGNORE_PATTERNS,
    ModelWeightsManager,
)


class FakeArtifactGateway(LLMArtifactGateway):
    """Minimal fake gateway for testing."""

    def __init__(self, existing_files: List[str] = None, uploaded: List[tuple] = None):
        self._existing_files = existing_files if existing_files is not None else []
        self.uploaded: List[tuple] = uploaded if uploaded is not None else []

    def list_files(self, path: str, **kwargs) -> List[str]:
        return self._existing_files

    def upload_files(self, local_path: str, remote_path: str, **kwargs) -> None:
        self.uploaded.append((local_path, remote_path))

    def download_files(self, path: str, target_path: str, overwrite=False, **kwargs) -> List[str]:
        return []

    def get_model_weights_urls(self, owner: str, model_name: str, **kwargs) -> List[str]:
        return []

    def get_model_config(self, path: str, **kwargs) -> Dict[str, Any]:
        return {}


@pytest.mark.asyncio
async def test_cache_hit_skips_download():
    """When list_files returns non-empty, no download or upload should occur."""
    gateway = FakeArtifactGateway(existing_files=["model.safetensors"])
    manager = ModelWeightsManager(llm_artifact_gateway=gateway)

    mwm_base = "model_engine_server.domain.use_cases.model_weights_manager"
    with (
        patch(f"{mwm_base}.snapshot_download") as mock_download,
        patch(f"{mwm_base}.asyncio.create_task") as mock_create_task,
    ):
        result = manager.ensure_model_weights_available("meta-llama/Meta-Llama-3-8B")
        # Run the background sync task to assert on side-effects
        await mock_create_task.call_args[0][0]

    mock_download.assert_not_called()
    assert len(gateway.uploaded) == 0
    assert "meta-llama/Meta-Llama-3-8B" in result


@pytest.mark.asyncio
async def test_cache_hit_returns_correct_s3_path(monkeypatch):
    """On cache hit the returned path should be {prefix}/{hf_repo}."""
    monkeypatch.setattr(
        "model_engine_server.domain.use_cases.model_weights_manager.hmi_config",
        MagicMock(hf_user_fine_tuned_weights_prefix="s3://my-bucket/weights"),
    )
    gateway = FakeArtifactGateway(existing_files=["file.bin"])
    manager = ModelWeightsManager(llm_artifact_gateway=gateway)

    mwm_base = "model_engine_server.domain.use_cases.model_weights_manager"
    with patch(f"{mwm_base}.asyncio.create_task") as mock_create_task:
        result = manager.ensure_model_weights_available("org/model")
        await mock_create_task.call_args[0][0]

    assert result == "s3://my-bucket/weights/org/model"


@pytest.mark.asyncio
async def test_cache_miss_calls_snapshot_download_and_upload(tmp_path, monkeypatch):
    """On cache miss, snapshot_download and upload_files should both be called."""
    monkeypatch.setattr(
        "model_engine_server.domain.use_cases.model_weights_manager.hmi_config",
        MagicMock(hf_user_fine_tuned_weights_prefix="s3://my-bucket/weights"),
    )

    gateway = FakeArtifactGateway(existing_files=[])
    manager = ModelWeightsManager(llm_artifact_gateway=gateway)

    mwm_base = "model_engine_server.domain.use_cases.model_weights_manager"
    with (
        patch(f"{mwm_base}.snapshot_download") as mock_download,
        patch(f"{mwm_base}.asyncio.create_task") as mock_create_task,
    ):
        result = manager.ensure_model_weights_available("org/model")
        # Run the background sync task so we can assert on its side-effects
        await mock_create_task.call_args[0][0]

    mock_download.assert_called_once()
    call_kwargs = mock_download.call_args
    assert call_kwargs.kwargs["repo_id"] == "org/model"
    assert call_kwargs.kwargs["ignore_patterns"] == HF_IGNORE_PATTERNS

    assert len(gateway.uploaded) == 1
    _local, remote = gateway.uploaded[0]
    assert remote == "s3://my-bucket/weights/org/model"
    assert result == "s3://my-bucket/weights/org/model"


def test_s3_path_construction(monkeypatch):
    """Remote path should be {prefix}/{hf_repo} with correct stripping of trailing slash."""
    monkeypatch.setattr(
        "model_engine_server.domain.use_cases.model_weights_manager.hmi_config",
        MagicMock(hf_user_fine_tuned_weights_prefix="s3://bucket/prefix/"),
    )
    gateway = FakeArtifactGateway(existing_files=[])
    manager = ModelWeightsManager(llm_artifact_gateway=gateway)

    path = manager.get_remote_path("myorg/mymodel")
    assert path == "s3://bucket/prefix/myorg/mymodel"


@pytest.mark.asyncio
async def test_create_llm_model_endpoint_calls_weights_manager_on_hf_source():
    """CreateLLMModelEndpointV1UseCase should call ensure_model_weights_available (sync),
    which returns the expected checkpoint path immediately and fires weight sync in the
    background. All following actions (bundle, endpoint creation) proceed without blocking."""
    from model_engine_server.domain.entities import LLMSource
    from model_engine_server.domain.use_cases.model_weights_manager import ModelWeightsManager

    mock_manager = MagicMock(spec=ModelWeightsManager)
    mock_manager.ensure_model_weights_available.return_value = (
        "s3://bucket/weights/huggyllama/llama-7b"
    )

    # Use a real SUPPORTED_MODELS_INFO entry: "llama-2-7b" -> "huggyllama/llama-7b"
    from tests.unit.conftest import FakeLLMArtifactGateway

    fake_gateway = FakeLLMArtifactGateway()

    from model_engine_server.domain.use_cases.llm_model_endpoint_use_cases import (
        CreateLLMModelEndpointV1UseCase,
    )

    mock_bundle_use_case = MagicMock()
    mock_bundle = MagicMock()
    mock_bundle.id = "bundle-id"
    mock_bundle_use_case.execute = AsyncMock(return_value=mock_bundle)

    mock_endpoint_service = MagicMock()
    mock_endpoint_record = MagicMock()
    mock_endpoint_record.id = "endpoint-id"
    mock_endpoint_record.creation_task_id = "task-123"
    mock_endpoint_service.create_model_endpoint = AsyncMock(return_value=mock_endpoint_record)
    mock_endpoint_service.can_scale_http_endpoint_from_zero = MagicMock(return_value=False)
    mock_endpoint_service.get_inference_autoscaling_metrics_gateway.return_value.emit_prewarm_metric = (
        AsyncMock()
    )

    mock_docker_repository = MagicMock()

    use_case = CreateLLMModelEndpointV1UseCase(
        create_llm_model_bundle_use_case=mock_bundle_use_case,
        model_endpoint_service=mock_endpoint_service,
        docker_repository=mock_docker_repository,
        llm_artifact_gateway=fake_gateway,
        model_weights_manager=mock_manager,
    )

    from model_engine_server.common.dtos.llms import CreateLLMModelEndpointV1Request
    from model_engine_server.core.auth.authentication_repository import User
    from pydantic import TypeAdapter

    user = User(user_id="test-user", team_id="test-team", is_privileged_user=True)
    request = TypeAdapter(CreateLLMModelEndpointV1Request).validate_python(
        {
            "name": "test-endpoint",
            "model_name": "llama-2-7b",
            "source": LLMSource.HUGGING_FACE,
            "inference_framework": "vllm",
            "inference_framework_image_tag": "0.1.0",
            "num_shards": 1,
            "endpoint_type": "streaming",
            "checkpoint_path": None,
            "min_workers": 1,
            "max_workers": 1,
            "per_worker": 10,
            "cpus": 4,
            "memory": "16Gi",
            "storage": "50Gi",
            "gpus": 1,
            "gpu_type": "nvidia-ampere-a10",
            "nodes_per_worker": 1,
            "labels": {"team": "test"},
            "metadata": {},
        }
    )

    # Patch infrastructure helpers to keep the test focused on weights manager behavior
    base = "model_engine_server.domain.use_cases.llm_model_endpoint_use_cases"
    with (
        patch(f"{base}._fill_hardware_info"),
        patch(f"{base}.validate_resource_requests"),
        patch(f"{base}.validate_deployment_resources"),
        patch(f"{base}.validate_labels"),
        patch(f"{base}.validate_billing_tags"),
        patch(f"{base}.validate_post_inference_hooks"),
        patch(f"{base}.validate_model_name"),
        patch(f"{base}.validate_num_shards"),
        patch(f"{base}.validate_quantization"),
        patch(f"{base}.validate_chat_template"),
        patch(f"{base}.LiveAuthorizationModule") as mock_authz,
    ):
        mock_authz.return_value.get_aws_role_for_user = MagicMock(
            return_value="arn:aws:iam::123:role/test"
        )
        mock_authz.return_value.get_s3_bucket_for_user = MagicMock(return_value="test-bucket")
        await use_case.execute(user=user, request=request)

    # ensure_model_weights_available is called synchronously — no await, no blocking
    mock_manager.ensure_model_weights_available.assert_called_once_with("huggyllama/llama-7b")
    # Verify that the resolved checkpoint path was forwarded to the bundle use case
    bundle_call_kwargs = mock_bundle_use_case.execute.call_args.kwargs
    assert bundle_call_kwargs["checkpoint_path"] == "s3://bucket/weights/huggyllama/llama-7b"
