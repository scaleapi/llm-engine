from typing import Dict

from huggingface_hub import list_repo_refs
from huggingface_hub.utils._errors import RepositoryNotFoundError
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


def load_tokenizer_from_s3(s3_repo: str, llm_artifact_gateway: LLMArtifactGateway) -> None:
    """
    Download tokenizer files from S3 to the local filesystem.
    """
    if not s3_repo:
        return

    for file in TOKENIZER_FILES_REQUIRED:
        s3_path = f"{s3_repo}/{file}"
        target_path = f"{TOKENIZER_TARGET_DIR}/{file}"
        llm_artifact_gateway.download_files(s3_path, target_path)

    for file in TOKENIZER_FILES_OPTIONAL:
        s3_path = f"{s3_repo}/{file}"
        target_path = f"{TOKENIZER_TARGET_DIR}/{file}"
        try:
            llm_artifact_gateway.download_files(s3_path, target_path)
        except Exception:  # noqa
            pass


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
        load_tokenizer_from_s3(model_info.s3_repo, llm_artifact_gateway)
        model_location = model_info.s3_repo

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
