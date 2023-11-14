from typing import Dict

from huggingface_hub import list_repo_refs
from huggingface_hub.utils._errors import RepositoryNotFoundError
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import ObjectNotFoundException
from model_engine_server.domain.gateways.llm_artifact_gateway import LLMArtifactGateway
from model_engine_server.domain.use_cases.llm_model_endpoint_use_cases import _SUPPORTED_MODELS_INFO
from transformers import AutoTokenizer

logger = make_logger(logger_name())


# Hack to count prompt tokens
tokenizer_cache: Dict[str, AutoTokenizer] = {}


TOKENIZER_FILES_REQUIRED = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]
TOKENIZER_FILES_OPTIONAL = [
    "tokenizer.model",
]
TOKENIZER_TARGET_DIR = "/root/.cache/model_engine_server/tokenizers"


def get_models_s3_prefix(model_prefix: str) -> str:
    """
    Get the S3 prefix for a given model prefix.
    """
    return f"models/{model_prefix}"


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
    model_name: str, s3_prefix: str, llm_artifact_gateway: LLMArtifactGateway
) -> str:
    """
    Download tokenizer files from S3 to the local filesystem.
    """
    if not s3_prefix:
        return ""

    for file in TOKENIZER_FILES_REQUIRED:
        s3_path = get_models_s3_uri(s3_prefix, file)
        target_path = get_models_local_path(model_name, file)
        llm_artifact_gateway.download_files(s3_path, target_path)

    for file in TOKENIZER_FILES_OPTIONAL:
        s3_path = get_models_s3_uri(s3_prefix, file)
        target_path = get_models_local_path(model_name, file)
        try:
            llm_artifact_gateway.download_files(s3_path, target_path)
        except Exception:  # noqa
            pass

    return f"{TOKENIZER_TARGET_DIR}/{model_name}"


def load_tokenizer(model_name: str, llm_artifact_gateway: LLMArtifactGateway) -> None:
    logger.info(f"Loading tokenizer for model {model_name}.")

    model_info = _SUPPORTED_MODELS_INFO[model_name]
    model_location = ""
    try:
        if not model_info.hf_repo:
            raise RepositoryNotFoundError("No HF repo specified for model.")
        list_repo_refs(model_info.hf_repo)  # check if model exists in Hugging Face Hub
        model_location = model_info.hf_repo
        # AutoTokenizer handles file downloads for HF repos
    except RepositoryNotFoundError as e:
        logger.warn(f"No HF repo for model {model_name} - {e}.")
        model_location = load_tokenizer_from_s3(
            model_name, model_info.s3_repo, llm_artifact_gateway
        )

    if not model_location:
        raise ObjectNotFoundException(f"Tokenizer not found for model {model_name}.")

    logger.info(f"Loading tokenizer for model {model_name} from {model_location}.")
    tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_location)


def get_tokenizer(model_name: str, llm_artifact_gateway: LLMArtifactGateway) -> AutoTokenizer:
    """
    Get tokenizer for a given model name and inference framework.
    """
    if model_name not in tokenizer_cache:
        load_tokenizer(model_name, llm_artifact_gateway)
    return tokenizer_cache[model_name]


def count_tokens(input: str, model_name: str, llm_artifact_gateway: LLMArtifactGateway) -> int:
    """
    Count the number of tokens in the input string.
    """
    tokenizer = get_tokenizer(model_name, llm_artifact_gateway)
    return len(tokenizer.encode(input))
