import os
from collections import namedtuple
from functools import lru_cache
from typing import Dict, Optional

from huggingface_hub import list_repo_refs
from huggingface_hub.utils._errors import RepositoryNotFoundError
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import ObjectNotFoundException
from model_engine_server.domain.gateways.llm_artifact_gateway import LLMArtifactGateway
from model_engine_server.domain.repositories.tokenizer_repository import TokenizerRepository
from transformers import AutoTokenizer

logger = make_logger(logger_name())


TOKENIZER_FILES_REQUIRED = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]
TOKENIZER_FILES_OPTIONAL = [
    "tokenizer.model",
]
TOKENIZER_TARGET_DIR = "/opt/.cache/model_engine_server/tokenizers"


ModelInfo = namedtuple("ModelInfo", ["hf_repo", "s3_repo"])


def get_default_supported_models_info() -> Dict[str, ModelInfo]:
    return {
        "mpt-7b": ModelInfo("mosaicml/mpt-7b", None),
        "mpt-7b-instruct": ModelInfo("mosaicml/mpt-7b-instruct", None),
        "flan-t5-xxl": ModelInfo("google/flan-t5-xxl", None),
        "llama-7b": ModelInfo("decapoda-research/llama-7b-hf", None),
        "llama-2-7b": ModelInfo("huggyllama/llama-7b", None),
        "llama-2-7b-chat": ModelInfo("meta-llama/Llama-2-7b-chat-hf", None),
        "llama-2-13b": ModelInfo("meta-llama/Llama-2-13b-hf", None),
        "llama-2-13b-chat": ModelInfo("meta-llama/Llama-2-13b-chat-hf", None),
        "llama-2-70b": ModelInfo("meta-llama/Llama-2-70b-hf", None),
        "llama-2-70b-chat": ModelInfo("meta-llama/Llama-2-70b-chat-hf", None),
        "falcon-7b": ModelInfo("tiiuae/falcon-7b", None),
        "falcon-7b-instruct": ModelInfo("tiiuae/falcon-7b-instruct", None),
        "falcon-40b": ModelInfo("tiiuae/falcon-40b", None),
        "falcon-40b-instruct": ModelInfo("tiiuae/falcon-40b-instruct", None),
        "falcon-180b": ModelInfo("tiiuae/falcon-180B", None),
        "falcon-180b-chat": ModelInfo("tiiuae/falcon-180B-chat", None),
        "codellama-7b": ModelInfo("codellama/CodeLlama-7b-hf", None),
        "codellama-7b-instruct": ModelInfo("codellama/CodeLlama-7b-Instruct-hf", None),
        "codellama-13b": ModelInfo("codellama/CodeLlama-13b-hf", None),
        "codellama-13b-instruct": ModelInfo("codellama/CodeLlama-13b-Instruct-hf", None),
        "codellama-34b": ModelInfo("codellama/CodeLlama-34b-hf", None),
        "codellama-34b-instruct": ModelInfo("codellama/CodeLlama-34b-Instruct-hf", None),
        "llm-jp-13b-instruct-full": ModelInfo("llm-jp/llm-jp-13b-instruct-full-jaster-v1.0", None),
        "llm-jp-13b-instruct-full-dolly": ModelInfo(
            "llm-jp/llm-jp-13b-instruct-full-dolly-oasst-v1.0", None
        ),
        "mistral-7b": ModelInfo("mistralai/Mistral-7B-v0.1", None),
        "mistral-7b-instruct": ModelInfo("mistralai/Mistral-7B-Instruct-v0.1", None),
        "mammoth-coder-llama-2-7b": ModelInfo("TIGER-Lab/MAmmoTH-Coder-7B", None),
        "mammoth-coder-llama-2-13b": ModelInfo("TIGER-Lab/MAmmoTH-Coder-13B", None),
        "mammoth-coder-llama-2-34b": ModelInfo("TIGER-Lab/MAmmoTH-Coder-34B", None),
        "gpt-j-6b": ModelInfo("EleutherAI/gpt-j-6b", None),
        "gpt-j-6b-zh-en": ModelInfo("EleutherAI/gpt-j-6b", None),
        "gpt4all-j": ModelInfo("nomic-ai/gpt4all-j", None),
        "dolly-v2-12b": ModelInfo("databricks/dolly-v2-12b", None),
        "stablelm-tuned-7b": ModelInfo("StabilityAI/stablelm-tuned-alpha-7b", None),
        "vicuna-13b": ModelInfo("eachadea/vicuna-13b-1.1", None),
        "zephyr-7b-alpha": ModelInfo("HuggingFaceH4/zephyr-7b-alpha", None),
        "zephyr-7b-beta": ModelInfo("HuggingFaceH4/zephyr-7b-beta", None),
    }


def get_supported_models_info() -> Dict[str, ModelInfo]:
    try:
        from plugins.live_tokenizer_repository import (
            get_supported_models_info as get_custom_supported_models_info,
        )

        return get_custom_supported_models_info()
    except ModuleNotFoundError:
        return get_default_supported_models_info()


SUPPORTED_MODELS_INFO = get_supported_models_info()


def get_models_s3_uri(*args, **kwargs) -> str:
    try:
        from plugins.live_tokenizer_repository import get_models_s3_uri as get_custom_models_s3_uri

        return get_custom_models_s3_uri(*args, **kwargs)
    except ModuleNotFoundError:
        raise NotImplementedError


def get_models_local_dir_path(model_name: str) -> str:
    """
    Get the local directory path for a given model.
    """
    return f"{TOKENIZER_TARGET_DIR}/{model_name}"


class LiveTokenizerRepository(TokenizerRepository):
    def __init__(self, llm_artifact_gateway: LLMArtifactGateway):
        self.llm_artifact_gateway = llm_artifact_gateway

    def _load_tokenizer_from_s3(self, model_name: str, s3_prefix: Optional[str]) -> Optional[str]:
        """
        Download tokenizer files from S3 to the local filesystem.
        """
        if not s3_prefix:
            return None

        model_tokenizer_dir = get_models_local_dir_path(model_name)

        for file in TOKENIZER_FILES_REQUIRED:
            s3_path = get_models_s3_uri(s3_prefix, file)
            target_path = os.path.join(model_tokenizer_dir, file)
            self.llm_artifact_gateway.download_files(s3_path, target_path)

        for file in TOKENIZER_FILES_OPTIONAL:
            s3_path = get_models_s3_uri(s3_prefix, file)
            target_path = os.path.join(model_tokenizer_dir, file)
            try:
                self.llm_artifact_gateway.download_files(s3_path, target_path)
            except Exception:
                pass

        return model_tokenizer_dir

    @lru_cache(maxsize=32)
    def load_tokenizer(self, model_name: str) -> AutoTokenizer:
        model_info = SUPPORTED_MODELS_INFO[model_name]

        model_location = None
        try:
            if not model_info.hf_repo:
                raise RepositoryNotFoundError("No HF repo specified for model.")
            list_repo_refs(model_info.hf_repo)  # check if model exists in Hugging Face Hub
            model_location = model_info.hf_repo
            # AutoTokenizer handles file downloads for HF repos
        except RepositoryNotFoundError:
            model_location = self._load_tokenizer_from_s3(model_name, model_info.s3_repo)

        if not model_location:
            raise ObjectNotFoundException(f"Tokenizer not found for model {model_name}.")

        logger.info(f"Loading tokenizer for model {model_name} from {model_location}.")
        return AutoTokenizer.from_pretrained(model_location)
