import asyncio
import functools
import tempfile
from typing import Dict, List, Set

from huggingface_hub import snapshot_download
from model_engine_server.common.config import hmi_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.gateways.llm_artifact_gateway import LLMArtifactGateway

logger = make_logger(logger_name())

# Match the internal sync_model_weights.py inclusion/exclusion patterns
HF_IGNORE_PATTERNS: List[str] = [
    "optimizer*",
    "*.msgpack",
    "*.h5",
    "flax_model*",
    "tf_model*",
    "rust_model*",
]


class ModelWeightsManager:
    def __init__(self, llm_artifact_gateway: LLMArtifactGateway):
        self.llm_artifact_gateway = llm_artifact_gateway
        self._background_tasks: Set[asyncio.Task] = set()
        self._in_progress: Dict[str, asyncio.Task] = {}

    def get_remote_path(self, hf_repo: str) -> str:
        prefix = hmi_config.hf_user_fine_tuned_weights_prefix.rstrip("/")
        return f"{prefix}/{hf_repo}"

    def ensure_model_weights_available(self, hf_repo: str) -> str:
        """
        Returns the expected remote path for ``hf_repo`` immediately and starts
        syncing weights from HuggingFace Hub to that path in the background.

        If the weights are already cached the background task exits early.
        Callers receive the checkpoint path right away and can proceed with
        any following actions (e.g. endpoint creation) without blocking.

        A second call for the same ``hf_repo`` while a sync is already in
        progress is a no-op: the existing task is reused and the same remote
        path is returned.

        Args:
            hf_repo: HuggingFace repository ID, e.g. ``"meta-llama/Meta-Llama-3-8B"``.

        Returns:
            The remote path (s3://, gs://, or https://) where the weights will be stored.
        """
        remote_path = self.get_remote_path(hf_repo)
        if hf_repo not in self._in_progress:
            task = asyncio.create_task(self._sync_weights(hf_repo, remote_path))
            self._background_tasks.add(task)
            self._in_progress[hf_repo] = task
            task.add_done_callback(lambda t: self._on_task_done(t, hf_repo))
        return remote_path

    def _on_task_done(self, task: asyncio.Task, hf_repo: str) -> None:
        self._background_tasks.discard(task)
        self._in_progress.pop(hf_repo, None)
        if not task.cancelled():
            exc = task.exception()
            if exc:
                logger.error(
                    f"Background weight sync failed for {hf_repo}: {exc}",
                    exc_info=exc,
                )

    async def _sync_weights(self, hf_repo: str, remote_path: str) -> None:
        """Downloads weights from HuggingFace Hub and uploads to remote storage if not cached."""
        files = self.llm_artifact_gateway.list_files(remote_path)
        if files:
            logger.info(f"Cache hit: {len(files)} files at {remote_path}")
            return

        logger.info(f"Cache miss for {hf_repo}. Downloading from HuggingFace Hub...")
        loop = asyncio.get_event_loop()
        with tempfile.TemporaryDirectory() as tmp_dir:
            await loop.run_in_executor(
                None,
                functools.partial(
                    snapshot_download,
                    repo_id=hf_repo,
                    local_dir=tmp_dir,
                    ignore_patterns=HF_IGNORE_PATTERNS,
                ),
            )
            await loop.run_in_executor(
                None,
                functools.partial(
                    self.llm_artifact_gateway.upload_files,
                    tmp_dir,
                    remote_path,
                ),
            )

        logger.info(f"Weights for {hf_repo} uploaded to {remote_path}")
