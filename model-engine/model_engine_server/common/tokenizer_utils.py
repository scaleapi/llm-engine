from collections import namedtuple
from functools import lru_cache
from typing import Optional

from huggingface_hub import list_repo_refs
from huggingface_hub.utils._errors import RepositoryNotFoundError
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import ObjectNotFoundException
from model_engine_server.domain.gateways.llm_artifact_gateway import LLMArtifactGateway
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


def get_models_s3_prefix(model_prefix: str) -> str:
    """
    Get the S3 prefix for a given model prefix.
    """
    return f"models/{model_prefix}"


ModelInfo = namedtuple("ModelInfo", ["hf_repo", "s3_repo"])

_SUPPORTED_MODELS_INFO = {
    "mpt-7b": ModelInfo("mosaicml/mpt-7b", None),
    "mpt-7b-instruct": ModelInfo("mosaicml/mpt-7b-instruct", None),
    "flan-t5-xxl": ModelInfo("google/flan-t5-xxl", None),
    "llama-7b": ModelInfo(
        "decapoda-research/llama-7b-hf", get_models_s3_prefix("hf-llama/hf-llama-7b")
    ),
    "llama-2-7b": ModelInfo(
        "meta-llama/Llama-2-7b-hf", get_models_s3_prefix("hf-llama/hf-llama-2-7b")
    ),
    "llama-2-7b-chat": ModelInfo(
        "meta-llama/Llama-2-7b-chat-hf", get_models_s3_prefix("hf-llama/hf-llama-2-7b-chat")
    ),
    "llama-2-13b": ModelInfo(
        "meta-llama/Llama-2-13b-hf", get_models_s3_prefix("hf-llama/hf-llama-2-13b")
    ),
    "llama-2-13b-chat": ModelInfo(
        "meta-llama/Llama-2-13b-chat-hf", get_models_s3_prefix("hf-llama/hf-llama-2-13b-chat")
    ),
    "llama-2-70b": ModelInfo(
        "meta-llama/Llama-2-70b-hf", get_models_s3_prefix("hf-llama/hf-llama-2-70b")
    ),
    "llama-2-70b-chat": ModelInfo(
        "meta-llama/Llama-2-70b-chat-hf", get_models_s3_prefix("hf-llama/hf-llama-2-70b-chat")
    ),
    "falcon-7b": ModelInfo("tiiuae/falcon-7b", None),
    "falcon-7b-instruct": ModelInfo("tiiuae/falcon-7b-instruct", None),
    "falcon-40b": ModelInfo("tiiuae/falcon-40b", None),
    "falcon-40b-instruct": ModelInfo("tiiuae/falcon-40b-instruct", None),
    "falcon-180b": ModelInfo("tiiuae/falcon-180B", get_models_s3_prefix("falcon-hf/falcon-180b")),
    "falcon-180b-chat": ModelInfo(
        "tiiuae/falcon-180B-chat", get_models_s3_prefix("falcon-hf/falcon-180b-chat")
    ),
    "codellama-7b": ModelInfo("codellama/CodeLlama-7b-hf", None),
    "codellama-7b-instruct": ModelInfo("codellama/CodeLlama-7b-Instruct-hf", None),
    "codellama-13b": ModelInfo("codellama/CodeLlama-13b-hf", None),
    "codellama-13b-instruct": ModelInfo("codellama/CodeLlama-13b-Instruct-hf", None),
    "codellama-34b": ModelInfo("codellama/CodeLlama-34b-hf", None),
    "codellama-34b-instruct": ModelInfo("codellama/CodeLlama-34b-Instruct-hf", None),
    "llm-jp-13b-instruct-full": ModelInfo(
        "llm-jp/llm-jp-13b-instruct-full-jaster-v1.0",
        get_models_s3_prefix("llm-jp/llm-jp-13b-instruct-full-jaster-v1.0"),
    ),
    "llm-jp-13b-instruct-full-dolly": ModelInfo(
        "llm-jp/llm-jp-13b-instruct-full-dolly-oasst-v1.0",
        get_models_s3_prefix("llm-jp/llm-jp--llm-jp-13b-instruct-full-dolly-oasst-v1.0"),
    ),
    "mistral-7b": ModelInfo("mistralai/Mistral-7B-v0.1", get_models_s3_prefix("mistral-7b")),
    "mistral-7b-instruct": ModelInfo(
        "mistralai/Mistral-7B-Instruct-v0.1", get_models_s3_prefix("mistral-7b-instruct")
    ),
    "mammoth-coder-llama-2-7b": ModelInfo(
        "TIGER-Lab/MAmmoTH-Coder-7B", get_models_s3_prefix("hf-llama/mammoth-coder-llama-2-7b")
    ),
    "mammoth-coder-llama-2-13b": ModelInfo(
        "TIGER-Lab/MAmmoTH-Coder-13B", get_models_s3_prefix("hf-llama/mammoth-coder-llama-2-13b")
    ),
    "mammoth-coder-llama-2-34b": ModelInfo(
        "TIGER-Lab/MAmmoTH-Coder-34B", get_models_s3_prefix("hf-llama/mammoth-coder-llama-2-34b")
    ),
    "gpt-j-6b": ModelInfo("EleutherAI/gpt-j-6b", None),
    "gpt-j-6b-zh-en": ModelInfo("EleutherAI/gpt-j-6b", None),
    "gpt4all-j": ModelInfo("nomic-ai/gpt4all-j", None),
    "dolly-v2-12b": ModelInfo("databricks/dolly-v2-12b", None),
    "stablelm-tuned-7b": ModelInfo("StabilityAI/stablelm-tuned-alpha-7b", None),
    "vicuna-13b": ModelInfo("eachadea/vicuna-13b-1.1", None),
}


def get_models_s3_uri(s3_prefix: str, file: str) -> str:
    """
    Get the S3 URI for a given model prefix and file.
    """
    return f"s3://{infra_config().s3_bucket}/{s3_prefix}/{file}"


def get_models_local_path(model_name: str, file: str) -> str:
    """
    Get the local path for a given model prefix and file.
    """
    return f"{TOKENIZER_TARGET_DIR}/{model_name}/{file}"


def load_tokenizer_from_s3(
    model_name: str, s3_prefix: Optional[str], llm_artifact_gateway: LLMArtifactGateway
) -> Optional[str]:
    """
    Download tokenizer files from S3 to the local filesystem.
    """
    if not s3_prefix:
        return None

    model_tokenizer_dir = f"{TOKENIZER_TARGET_DIR}/{model_name}"

    for file in TOKENIZER_FILES_REQUIRED:
        s3_path = get_models_s3_uri(s3_prefix, file)
        target_path = get_models_local_path(model_name, file)
        llm_artifact_gateway.download_files(s3_path, target_path)

    for file in TOKENIZER_FILES_OPTIONAL:
        s3_path = get_models_s3_uri(s3_prefix, file)
        target_path = get_models_local_path(model_name, file)
        try:
            llm_artifact_gateway.download_files(s3_path, target_path)
        except Exception:
            pass

    return model_tokenizer_dir


@lru_cache(maxsize=32)
def load_tokenizer(model_name: str, llm_artifact_gateway: LLMArtifactGateway) -> AutoTokenizer:
    model_info = _SUPPORTED_MODELS_INFO[model_name]
    model_location = None
    try:
        if not model_info.hf_repo:
            raise RepositoryNotFoundError("No HF repo specified for model.")
        list_repo_refs(model_info.hf_repo)  # check if model exists in Hugging Face Hub
        model_location = model_info.hf_repo
        # AutoTokenizer handles file downloads for HF repos
    except RepositoryNotFoundError:
        model_location = load_tokenizer_from_s3(
            model_name, model_info.s3_repo, llm_artifact_gateway
        )

    if not model_location:
        raise ObjectNotFoundException(f"Tokenizer not found for model {model_name}.")

    logger.info(f"Loading tokenizer for model {model_name} from {model_location}.")
    return AutoTokenizer.from_pretrained(model_location)


def count_tokens(input: str, model_name: str, llm_artifact_gateway: LLMArtifactGateway) -> int:
    """
    Count the number of tokens in the input string.
    """
    tokenizer = load_tokenizer(model_name, llm_artifact_gateway)
    return len(tokenizer.encode(input))
