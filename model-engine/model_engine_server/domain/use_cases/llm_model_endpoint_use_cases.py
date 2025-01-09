"""
TODO figure out how to do: (or if we want to do it)
List model endpoint history: GET model-endpoints/<endpoint id>/history
Read model endpoint creation logs: GET model-endpoints/<endpoint id>/creation-logs
"""

import base64
import datetime
import json
import math
import os
import re
from dataclasses import asdict
from functools import lru_cache
from typing import Any, AsyncGenerator, AsyncIterable, Dict, List, Optional, Union

import yaml
from model_engine_server.common.config import hmi_config
from model_engine_server.common.dtos.batch_jobs import CreateDockerImageBatchJobResourceRequests
from model_engine_server.common.dtos.llms import (
    ChatCompletionV2Request,
    ChatCompletionV2StreamSuccessChunk,
    ChatCompletionV2SyncResponse,
    CompletionOutput,
    CompletionStreamOutput,
    CompletionStreamV1Request,
    CompletionStreamV1Response,
    CompletionSyncV1Request,
    CompletionSyncV1Response,
    CreateBatchCompletionsEngineRequest,
    CreateBatchCompletionsV1Request,
    CreateBatchCompletionsV1Response,
    CreateBatchCompletionsV2Request,
    CreateBatchCompletionsV2Response,
    CreateLLMModelEndpointV1Request,
    CreateLLMModelEndpointV1Response,
    DeleteLLMEndpointResponse,
    GetLLMModelEndpointV1Response,
    ListLLMModelEndpointsV1Response,
    ModelDownloadRequest,
    ModelDownloadResponse,
    TokenOutput,
    UpdateLLMModelEndpointV1Request,
    UpdateLLMModelEndpointV1Response,
)
from model_engine_server.common.dtos.llms.batch_completion import (
    CancelBatchCompletionsV2Response,
    GetBatchCompletionV2Response,
    UpdateBatchCompletionsV2Request,
    UpdateBatchCompletionsV2Response,
)
from model_engine_server.common.dtos.llms.completion import (
    CompletionV2Request,
    CompletionV2StreamSuccessChunk,
    CompletionV2SyncResponse,
)
from model_engine_server.common.dtos.llms.vllm import VLLMEndpointAdditionalArgs, VLLMModelConfig
from model_engine_server.common.dtos.model_bundles import CreateModelBundleV2Request
from model_engine_server.common.dtos.model_endpoints import ModelEndpointOrderBy
from model_engine_server.common.dtos.tasks import SyncEndpointPredictV1Request, TaskStatus
from model_engine_server.common.resource_limits import validate_resource_requests
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.configmap import read_config_map
from model_engine_server.core.loggers import (
    LoggerTagKey,
    LoggerTagManager,
    logger_name,
    make_logger,
)
from model_engine_server.domain.entities import (
    GpuType,
    LLMInferenceFramework,
    LLMMetadata,
    LLMSource,
    ModelBundle,
    ModelBundleFlavorType,
    ModelEndpoint,
    ModelEndpointType,
    Quantization,
    RunnableImageFlavor,
    RunnableImageLike,
    StreamingEnhancedRunnableImageFlavor,
)
from model_engine_server.domain.entities.docker_image_batch_job_bundle_entity import (
    DockerImageBatchJobBundle,
)
from model_engine_server.domain.exceptions import (
    DockerImageNotFoundException,
    EndpointInfraStateNotFound,
    EndpointLabelsException,
    EndpointUnsupportedInferenceTypeException,
    EndpointUnsupportedRequestException,
    FailToInferHardwareException,
    InvalidRequestException,
    LatestImageTagNotFoundException,
    ObjectHasInvalidValueException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
    UpstreamServiceError,
)
from model_engine_server.domain.gateways import (
    DockerImageBatchJobGateway,
    StreamingModelEndpointInferenceGateway,
)
from model_engine_server.domain.gateways.llm_artifact_gateway import LLMArtifactGateway
from model_engine_server.domain.repositories import (
    DockerImageBatchJobBundleRepository,
    DockerRepository,
    ModelBundleRepository,
    TokenizerRepository,
)
from model_engine_server.domain.services import LLMModelEndpointService, ModelEndpointService
from model_engine_server.domain.services.llm_batch_completions_service import (
    LLMBatchCompletionsService,
)
from model_engine_server.infra.gateways.filesystem_gateway import FilesystemGateway
from model_engine_server.infra.repositories.live_tokenizer_repository import (
    SUPPORTED_MODELS_INFO,
    get_models_s3_uri,
)

from ...common.datadog_utils import add_trace_model_name, add_trace_request_id
from ..authorization.live_authorization_module import LiveAuthorizationModule
from .model_bundle_use_cases import CreateModelBundleV2UseCase
from .model_endpoint_use_cases import (
    CONVERTED_FROM_ARTIFACT_LIKE_KEY,
    _handle_post_inference_hooks,
    model_endpoint_entity_to_get_model_endpoint_response,
    validate_billing_tags,
    validate_deployment_resources,
    validate_labels,
    validate_post_inference_hooks,
)

logger = make_logger(logger_name())

OPENAI_CHAT_COMPLETION_PATH = "/v1/chat/completions"
CHAT_TEMPLATE_MAX_LENGTH = 10_000
CHAT_SUPPORTED_INFERENCE_FRAMEWORKS = [LLMInferenceFramework.VLLM]

OPENAI_COMPLETION_PATH = "/v1/completions"
OPENAI_SUPPORTED_INFERENCE_FRAMEWORKS = [LLMInferenceFramework.VLLM]

LLM_METADATA_KEY = "_llm"
RESERVED_METADATA_KEYS = [LLM_METADATA_KEY, CONVERTED_FROM_ARTIFACT_LIKE_KEY]
VLLM_MODEL_WEIGHTS_FOLDER = "model_files"

LLM_MAX_CONCURRENCY_PER_WORKER = 250
# TODO as of Dec 2024 sync concurrency settings aren't implemented through the API so this does nothing
# In any case, the "true" value is 200 but we should probably set it higher

INFERENCE_FRAMEWORK_REPOSITORY: Dict[LLMInferenceFramework, str] = {
    LLMInferenceFramework.DEEPSPEED: "instant-llm",
    LLMInferenceFramework.TEXT_GENERATION_INFERENCE: hmi_config.tgi_repository,
    LLMInferenceFramework.VLLM: hmi_config.vllm_repository,
    LLMInferenceFramework.LIGHTLLM: hmi_config.lightllm_repository,
    LLMInferenceFramework.TENSORRT_LLM: hmi_config.tensorrt_llm_repository,
}

_SUPPORTED_MODELS_BY_FRAMEWORK = {
    LLMInferenceFramework.DEEPSPEED: set(
        [
            "mpt-7b",
            "mpt-7b-instruct",
            "flan-t5-xxl",
            "llama-7b",
            "gpt-j-6b",
            "gpt-j-6b-zh-en",
            "gpt4all-j",
            "dolly-v2-12b",
            "stablelm-tuned-7b",
            "vicuna-13b",
        ]
    ),
    LLMInferenceFramework.TEXT_GENERATION_INFERENCE: set(
        [
            "mpt-7b",
            "mpt-7b-instruct",
            "flan-t5-xxl",
            "llama-7b",
            "llama-2-7b",
            "llama-2-7b-chat",
            "llama-2-13b",
            "llama-2-13b-chat",
            "llama-2-70b",
            "llama-2-70b-chat",
            "falcon-7b",
            "falcon-7b-instruct",
            "falcon-40b",
            "falcon-40b-instruct",
            "codellama-7b",
            "codellama-7b-instruct",
            "codellama-13b",
            "codellama-13b-instruct",
            "codellama-34b",
            "codellama-34b-instruct",
            "llm-jp-13b-instruct-full",
            "llm-jp-13b-instruct-full-dolly",
            "zephyr-7b-alpha",
            "zephyr-7b-beta",
        ]
    ),
    LLMInferenceFramework.VLLM: set(
        [
            "mpt-7b",
            "mpt-7b-instruct",
            "llama-7b",
            "llama-2-7b",
            "llama-2-7b-chat",
            "llama-2-13b",
            "llama-2-13b-chat",
            "llama-2-70b",
            "llama-2-70b-chat",
            "llama-3-8b",
            "llama-3-8b-instruct",
            "llama-3-8b-instruct-262k",
            "llama-3-70b",
            "llama-3-70b-instruct",
            "llama-3-1-8b",
            "llama-3-1-8b-instruct",
            "llama-3-1-70b",
            "llama-3-1-70b-instruct",
            "llama-3-1-405b",
            "llama-3-1-405b-instruct",
            "llama-3-2-1b-instruct",
            "llama-3-2-3b-instruct",
            "llama-3-2-11b-vision-instruct",
            "llama-3-2-90b-vision-instruct",
            "falcon-7b",
            "falcon-7b-instruct",
            "falcon-40b",
            "falcon-40b-instruct",
            "falcon-180b",
            "falcon-180b-chat",
            "codellama-7b",
            "codellama-7b-instruct",
            "codellama-13b",
            "codellama-13b-instruct",
            "codellama-34b",
            "codellama-34b-instruct",
            "codellama-70b",
            "codellama-70b-instruct",
            "mistral-7b",
            "mistral-7b-instruct",
            "mixtral-8x7b",
            "mixtral-8x7b-instruct",
            "mixtral-8x22b",
            "mixtral-8x22b-instruct",
            "mammoth-coder-llama-2-7b",
            "mammoth-coder-llama-2-13b",
            "mammoth-coder-llama-2-34b",
            "zephyr-7b-alpha",
            "zephyr-7b-beta",
            "gemma-2b",
            "gemma-2b-instruct",
            "gemma-7b",
            "gemma-7b-instruct",
            "phi-3-mini-4k-instruct",
            "phi-3-mini-128k-instruct",
            "phi-3-small-8k-instruct",
            "phi-3-small-128k-instruct",
            "phi-3-medium-4-instruct",
            "phi-3-medium-128k-instruct",
            "deepseek-v2",
            "deepseek-v2-chat",
            "deepseek-coder-v2",
            "deepseek-coder-v2-instruct",
            "deepseek-coder-v2-lite",
            "deepseek-coder-v2-lite-instruct",
            "qwen2-72b-instruct",
        ]
    ),
    LLMInferenceFramework.LIGHTLLM: set(
        [
            "llama-7b",
            "llama-2-7b",
            "llama-2-7b-chat",
            "llama-2-13b",
            "llama-2-13b-chat",
            "llama-2-70b",
            "llama-2-70b-chat",
        ]
    ),
    LLMInferenceFramework.TENSORRT_LLM: set(
        ["llama-2-7b", "mixtral-8x7b", "mixtral-8x7b-instruct"]
    ),
}

_SUPPORTED_QUANTIZATIONS: Dict[LLMInferenceFramework, List[Quantization]] = {
    LLMInferenceFramework.DEEPSPEED: [],
    LLMInferenceFramework.TEXT_GENERATION_INFERENCE: [Quantization.BITSANDBYTES],
    LLMInferenceFramework.VLLM: [Quantization.AWQ],
    LLMInferenceFramework.LIGHTLLM: [],
    LLMInferenceFramework.TENSORRT_LLM: [],
}


NUM_DOWNSTREAM_REQUEST_RETRIES = 80  # has to be high enough so that the retries take the 5 minutes
DOWNSTREAM_REQUEST_TIMEOUT_SECONDS = 5 * 60  # 5 minutes

SERVICE_NAME = "model-engine"
SERVICE_IDENTIFIER = os.getenv("SERVICE_IDENTIFIER")
if SERVICE_IDENTIFIER:
    SERVICE_NAME += f"-{SERVICE_IDENTIFIER}"
LATEST_INFERENCE_FRAMEWORK_CONFIG_MAP_NAME = f"{SERVICE_NAME}-inference-framework-latest-config"
RECOMMENDED_HARDWARE_CONFIG_MAP_NAME = f"{SERVICE_NAME}-recommended-hardware-config"


def count_tokens(input: str, model_name: str, tokenizer_repository: TokenizerRepository) -> int:
    """
    Count the number of tokens in the input string.
    """
    tokenizer = tokenizer_repository.load_tokenizer(model_name)
    return len(tokenizer.encode(input))


async def _get_latest_batch_v2_tag(inference_framework: LLMInferenceFramework) -> str:
    config_map = await read_config_map(LATEST_INFERENCE_FRAMEWORK_CONFIG_MAP_NAME)
    batch_key = f"{inference_framework}_batch_v2"
    if batch_key not in config_map:
        raise LatestImageTagNotFoundException(
            f"Could not find latest batch job tag for inference framework {inference_framework}. key: {batch_key}"
        )
    return config_map[batch_key]


async def _get_latest_batch_tag(inference_framework: LLMInferenceFramework) -> str:
    config_map = await read_config_map(LATEST_INFERENCE_FRAMEWORK_CONFIG_MAP_NAME)
    batch_key = f"{inference_framework}_batch"
    if batch_key not in config_map:
        raise LatestImageTagNotFoundException(
            f"Could not find latest batch job tag for inference framework {inference_framework}. key: {batch_key}"
        )
    return config_map[batch_key]


async def _get_latest_tag(inference_framework: LLMInferenceFramework) -> str:
    config_map = await read_config_map(LATEST_INFERENCE_FRAMEWORK_CONFIG_MAP_NAME)
    if inference_framework not in config_map:
        raise LatestImageTagNotFoundException(
            f"Could not find latest tag for inference framework {inference_framework}."
        )
    return config_map[inference_framework]


async def _get_recommended_hardware_config_map() -> Dict[str, Any]:
    try:
        config_map = await read_config_map(RECOMMENDED_HARDWARE_CONFIG_MAP_NAME)
    except Exception as e:
        logger.error(
            f"Failed to read config map {RECOMMENDED_HARDWARE_CONFIG_MAP_NAME}, can't infer hardware config."
        )
        raise FailToInferHardwareException(
            f"Failed to read config map {RECOMMENDED_HARDWARE_CONFIG_MAP_NAME}, can't infer hardware config."
        ) from e
    return config_map


def _model_endpoint_entity_to_get_llm_model_endpoint_response(
    model_endpoint: ModelEndpoint,
) -> GetLLMModelEndpointV1Response:
    if (
        model_endpoint.record.metadata is None
        or LLM_METADATA_KEY not in model_endpoint.record.metadata
    ):
        raise ObjectHasInvalidValueException(
            f"Can't translate model entity to response, endpoint {model_endpoint.record.id} does not have LLM metadata."
        )
    llm_metadata = model_endpoint.record.metadata.get(LLM_METADATA_KEY, {})
    response = GetLLMModelEndpointV1Response(
        id=model_endpoint.record.id,
        name=model_endpoint.record.name,
        model_name=llm_metadata["model_name"],
        source=llm_metadata["source"],
        status=model_endpoint.record.status,
        inference_framework=llm_metadata["inference_framework"],
        inference_framework_image_tag=llm_metadata["inference_framework_image_tag"],
        num_shards=llm_metadata["num_shards"],
        quantize=llm_metadata.get("quantize"),
        checkpoint_path=llm_metadata.get("checkpoint_path"),
        chat_template_override=llm_metadata.get("chat_template_override"),
        spec=model_endpoint_entity_to_get_model_endpoint_response(model_endpoint),
    )
    return response


def validate_model_name(model_name: str, inference_framework: LLMInferenceFramework) -> None:
    # TODO: replace this logic to check if the model architecture is supported instead
    if model_name not in _SUPPORTED_MODELS_BY_FRAMEWORK[inference_framework]:
        logger.warning(
            f"Model name {model_name} may not be supported by inference framework {inference_framework}."
        )


def validate_num_shards(
    num_shards: int, inference_framework: LLMInferenceFramework, gpus: int
) -> None:
    if inference_framework == LLMInferenceFramework.DEEPSPEED:
        if num_shards <= 1:
            raise ObjectHasInvalidValueException("DeepSpeed requires more than 1 GPU.")
        if num_shards != gpus:
            raise ObjectHasInvalidValueException(
                f"Num shard {num_shards} must be the same as number of GPUs {gpus} for DeepSpeed."
            )
    if num_shards != gpus:
        raise ObjectHasInvalidValueException(
            f"Num shard {num_shards} must be equal to the number of GPUs {gpus}."
        )


def validate_quantization(
    quantize: Optional[Quantization], inference_framework: LLMInferenceFramework
) -> None:
    if quantize is not None and quantize not in _SUPPORTED_QUANTIZATIONS[inference_framework]:
        raise ObjectHasInvalidValueException(
            f"Quantization {quantize} is not supported for inference framework {inference_framework}. Supported quantization types are {_SUPPORTED_QUANTIZATIONS[inference_framework]}."
        )


def validate_chat_template(
    chat_template: Optional[str], inference_framework: LLMInferenceFramework
) -> None:
    if chat_template is not None:
        if len(chat_template) > CHAT_TEMPLATE_MAX_LENGTH:
            raise ObjectHasInvalidValueException(
                f"Chat template length must be less than {CHAT_TEMPLATE_MAX_LENGTH}."
            )

        if inference_framework != LLMInferenceFramework.VLLM:
            raise ObjectHasInvalidValueException(
                f"Chat template is only supported for inference framework {LLMInferenceFramework.VLLM}."
            )


def validate_checkpoint_path_uri(checkpoint_path: str) -> None:
    if (
        not checkpoint_path.startswith("s3://")
        and not checkpoint_path.startswith("azure://")
        and "blob.core.windows.net" not in checkpoint_path
    ):
        raise ObjectHasInvalidValueException(
            f"Only S3 and Azure Blob Storage paths are supported. Given checkpoint path: {checkpoint_path}."
        )
    if checkpoint_path.endswith(".tar"):
        raise ObjectHasInvalidValueException(
            f"Tar files are not supported. Given checkpoint path: {checkpoint_path}."
        )


def get_checkpoint_path(model_name: str, checkpoint_path_override: Optional[str]) -> str:
    checkpoint_path = None
    models_info = SUPPORTED_MODELS_INFO.get(model_name, None)
    if checkpoint_path_override:
        checkpoint_path = checkpoint_path_override
    elif models_info and models_info.s3_repo:
        checkpoint_path = get_models_s3_uri(models_info.s3_repo, "")  # pragma: no cover

    if not checkpoint_path:
        raise InvalidRequestException(f"No checkpoint path found for model {model_name}")

    validate_checkpoint_path_uri(checkpoint_path)
    return checkpoint_path


def validate_checkpoint_files(checkpoint_files: List[str]) -> None:
    """Require safetensors in the checkpoint path."""
    model_files = [f for f in checkpoint_files if "model" in f]
    num_safetensors = len([f for f in model_files if f.endswith(".safetensors")])
    if num_safetensors == 0:
        raise ObjectHasInvalidValueException("No safetensors found in the checkpoint path.")


def encode_template(chat_template: str) -> str:
    """Base64 encode the chat template to safely pass it to bash."""

    encoded = base64.b64encode(chat_template.encode("utf-8")).decode("utf-8")
    return encoded


class CreateLLMModelBundleV1UseCase:
    def __init__(
        self,
        create_model_bundle_use_case: CreateModelBundleV2UseCase,
        model_bundle_repository: ModelBundleRepository,
        llm_artifact_gateway: LLMArtifactGateway,
        docker_repository: DockerRepository,
    ):
        self.authz_module = LiveAuthorizationModule()
        self.create_model_bundle_use_case = create_model_bundle_use_case
        self.model_bundle_repository = model_bundle_repository
        self.llm_artifact_gateway = llm_artifact_gateway
        self.docker_repository = docker_repository

    def check_docker_image_exists_for_image_tag(
        self, framework_image_tag: str, repository_name: str
    ):
        if not self.docker_repository.image_exists(
            image_tag=framework_image_tag,
            repository_name=repository_name,
        ):
            raise DockerImageNotFoundException(
                repository=repository_name,
                tag=framework_image_tag,
            )

    async def execute(
        self,
        user: User,
        endpoint_name: str,
        model_name: str,
        source: LLMSource,
        framework: LLMInferenceFramework,
        framework_image_tag: str,
        endpoint_type: ModelEndpointType,
        num_shards: int,
        quantize: Optional[Quantization],
        checkpoint_path: Optional[str],
        chat_template_override: Optional[str],
        nodes_per_worker: int,
        additional_args: Optional[Dict[str, Any]] = None,
    ) -> ModelBundle:
        multinode = nodes_per_worker > 1
        if source == LLMSource.HUGGING_FACE:
            self.check_docker_image_exists_for_image_tag(
                framework_image_tag, INFERENCE_FRAMEWORK_REPOSITORY[framework]
            )
            if multinode and framework != LLMInferenceFramework.VLLM:
                raise ObjectHasInvalidValueException(
                    f"Multinode is not supported for framework {framework}."
                )

            if framework == LLMInferenceFramework.DEEPSPEED:
                bundle_id = await self.create_deepspeed_bundle(
                    user,
                    model_name,
                    framework_image_tag,
                    endpoint_type,
                    endpoint_name,
                )
            elif framework == LLMInferenceFramework.TEXT_GENERATION_INFERENCE:
                bundle_id = await self.create_text_generation_inference_bundle(
                    user,
                    model_name,
                    framework_image_tag,
                    endpoint_name,
                    num_shards,
                    quantize,
                    checkpoint_path,
                )
            elif framework == LLMInferenceFramework.VLLM:
                additional_vllm_args = (
                    VLLMEndpointAdditionalArgs.model_validate(additional_args)
                    if additional_args
                    else None
                )
                if multinode:
                    bundle_id = await self.create_vllm_multinode_bundle(
                        user,
                        model_name,
                        framework_image_tag,
                        endpoint_name,
                        num_shards,
                        nodes_per_worker,
                        quantize,
                        checkpoint_path,
                        chat_template_override,
                        additional_args=additional_vllm_args,
                    )
                else:
                    bundle_id = await self.create_vllm_bundle(
                        user,
                        model_name,
                        framework_image_tag,
                        endpoint_name,
                        num_shards,
                        quantize,
                        checkpoint_path,
                        chat_template_override,
                        additional_args=additional_vllm_args,
                    )
            elif framework == LLMInferenceFramework.LIGHTLLM:
                bundle_id = await self.create_lightllm_bundle(
                    user,
                    model_name,
                    framework_image_tag,
                    endpoint_name,
                    num_shards,
                    checkpoint_path,
                )
            elif framework == LLMInferenceFramework.TENSORRT_LLM:
                bundle_id = await self.create_tensorrt_llm_bundle(
                    user,
                    framework_image_tag,
                    endpoint_name,
                    num_shards,
                    checkpoint_path,
                )
            else:
                raise ObjectHasInvalidValueException(
                    f"Framework {framework} is not supported for source {source}."
                )
        else:
            raise ObjectHasInvalidValueException(f"Source {source} is not supported.")

        model_bundle = await self.model_bundle_repository.get_model_bundle(bundle_id)
        if model_bundle is None:
            raise ObjectNotFoundException(f"Model bundle {bundle_id} was not found after creation.")
        return model_bundle

    async def create_text_generation_inference_bundle(
        self,
        user: User,
        model_name: str,
        framework_image_tag: str,
        endpoint_unique_name: str,
        num_shards: int,
        quantize: Optional[Quantization],
        checkpoint_path: Optional[str],
    ):
        command = []

        # TGI requires max_input_length < max_total_tokens
        max_input_length = 1024
        max_total_tokens = 2048
        if "llama-2" in model_name:
            max_input_length = 4095
            max_total_tokens = 4096

        subcommands = []

        checkpoint_path = get_checkpoint_path(model_name, checkpoint_path)
        final_weights_folder = "model_files"

        subcommands += self.load_model_weights_sub_commands(
            LLMInferenceFramework.TEXT_GENERATION_INFERENCE,
            framework_image_tag,
            checkpoint_path,
            final_weights_folder,
        )

        subcommands.append(
            f"text-generation-launcher --hostname :: --model-id {final_weights_folder}  --num-shard {num_shards} --port 5005 --max-input-length {max_input_length} --max-total-tokens {max_total_tokens}"
        )

        if quantize:
            subcommands[-1] = subcommands[-1] + f" --quantize {quantize}"
        command = [
            "/bin/bash",
            "-c",
            ";".join(subcommands),
        ]

        return (
            await self.create_model_bundle_use_case.execute(
                user,
                CreateModelBundleV2Request(
                    name=endpoint_unique_name,
                    schema_location="TBA",
                    flavor=StreamingEnhancedRunnableImageFlavor(
                        flavor=ModelBundleFlavorType.STREAMING_ENHANCED_RUNNABLE_IMAGE,
                        repository=hmi_config.tgi_repository,
                        tag=framework_image_tag,
                        command=command,
                        streaming_command=command,
                        protocol="http",
                        readiness_initial_delay_seconds=10,
                        healthcheck_route="/health",
                        predict_route="/generate",
                        streaming_predict_route="/generate_stream",
                        env={},
                    ),
                    metadata={},
                ),
                do_auth_check=False,
                # Skip auth check because llm create endpoint is called as the user itself,
                # but the user isn't directly making the action. It should come from the fine tune
                # job.
            )
        ).model_bundle_id

    def load_model_weights_sub_commands(
        self,
        framework,
        framework_image_tag,
        checkpoint_path,
        final_weights_folder,
        trust_remote_code: bool = False,
    ):
        if checkpoint_path.startswith("s3://"):
            return self.load_model_weights_sub_commands_s3(
                framework,
                framework_image_tag,
                checkpoint_path,
                final_weights_folder,
                trust_remote_code,
            )
        elif checkpoint_path.startswith("azure://") or "blob.core.windows.net" in checkpoint_path:
            return self.load_model_weights_sub_commands_abs(
                framework,
                framework_image_tag,
                checkpoint_path,
                final_weights_folder,
                trust_remote_code,
            )
        else:
            raise ObjectHasInvalidValueException(
                f"Only S3 and Azure Blob Storage paths are supported. Given checkpoint path: {checkpoint_path}."
            )

    def load_model_weights_sub_commands_s3(
        self,
        framework,
        framework_image_tag,
        checkpoint_path,
        final_weights_folder,
        trust_remote_code: bool,
    ):
        subcommands = []
        s5cmd = "s5cmd"

        # This is a hack for now to skip installing s5cmd for text-generation-inference:0.9.3-launch_s3,
        # which has s5cmd binary already baked in. Otherwise, install s5cmd if it's not already available
        if (
            framework == LLMInferenceFramework.TEXT_GENERATION_INFERENCE
            and framework_image_tag != "0.9.3-launch_s3"
        ):
            subcommands.append(f"{s5cmd} > /dev/null || conda install -c conda-forge -y {s5cmd}")
        else:
            s5cmd = "./s5cmd"

        checkpoint_files = self.llm_artifact_gateway.list_files(checkpoint_path)
        validate_checkpoint_files(checkpoint_files)

        # filter to configs ('*.model' and '*.json') and weights ('*.safetensors')
        # For models that are not supported by transformers directly, we need to include '*.py' and '*.bin'
        # to load the model. Only set this flag if "trust_remote_code" is set to True
        file_selection_str = '--include "*.model" --include "*.model.v*" --include "*.json" --include "*.safetensors" --exclude "optimizer*"'
        if trust_remote_code:
            file_selection_str += ' --include "*.py"'
        subcommands.append(
            f"{s5cmd} --numworkers 512 cp --concurrency 10 {file_selection_str} {os.path.join(checkpoint_path, '*')} {final_weights_folder}"
        )
        return subcommands

    def load_model_weights_sub_commands_abs(
        self,
        framework,
        framework_image_tag,
        checkpoint_path,
        final_weights_folder,
        trust_remote_code: bool,
    ):
        subcommands = []

        subcommands.extend(
            [
                "export AZCOPY_AUTO_LOGIN_TYPE=WORKLOAD",
                "curl -L https://aka.ms/downloadazcopy-v10-linux | tar --strip-components=1 -C /usr/local/bin --no-same-owner --exclude=*.txt -xzvf - && chmod 755 /usr/local/bin/azcopy",
            ]
        )

        base_path = checkpoint_path.split("/")[-1]
        if base_path.endswith(".tar"):
            # If the checkpoint file is a tar file, extract it into final_weights_folder
            subcommands.extend(
                [
                    f"azcopy copy {checkpoint_path} .",
                    f"mkdir -p {final_weights_folder}",
                    f"tar --no-same-owner -xf {base_path} -C {final_weights_folder}",
                ]
            )
        else:
            additional_pattern = ";*.py" if trust_remote_code else ""
            file_selection_str = f'--include-pattern "*.model;*.json;*.safetensors{additional_pattern}" --exclude-pattern "optimizer*"'
            subcommands.append(
                f"azcopy copy --recursive {file_selection_str} {os.path.join(checkpoint_path, '*')} {final_weights_folder}"
            )

        return subcommands

    def load_model_files_sub_commands_trt_llm(
        self,
        checkpoint_path,
    ):
        """
        This function generate subcommands to load model files for TensorRT-LLM.
        Each model checkpoint is constituted of two folders: `model_weights` which stores the model engine files,
        and `model_tokenizer` which stores the model tokenizer files.
        See llm-engine/model-engine/model_engine_server/inference/tensorrt-llm/triton_model_repo/tensorrt_llm/config.pbtxt
        and llm-engine/model-engine/model_engine_server/inference/tensorrt-llm/triton_model_repo/postprocessing/config.pbtxt
        """
        if checkpoint_path.startswith("s3://"):
            subcommands = [
                f"./s5cmd --numworkers 512 cp --concurrency 50 {os.path.join(checkpoint_path, '*')} ./"
            ]
        else:
            subcommands.extend(
                [
                    "export AZCOPY_AUTO_LOGIN_TYPE=WORKLOAD",
                    "curl -L https://aka.ms/downloadazcopy-v10-linux | tar --strip-components=1 -C /usr/local/bin --no-same-owner --exclude=*.txt -xzvf - && chmod 755 /usr/local/bin/azcopy",
                    f"azcopy copy --recursive {os.path.join(checkpoint_path, '*')} ./",
                ]
            )
        return subcommands

    async def create_deepspeed_bundle(
        self,
        user: User,
        model_name: str,
        framework_image_tag: str,
        endpoint_type: ModelEndpointType,
        endpoint_unique_name: str,
    ):
        if endpoint_type == ModelEndpointType.STREAMING:
            command = [
                "dumb-init",
                "--",
                "ddtrace-run",
                "run-streamer",
                "--http",
                "production_threads",
                "--concurrency",
                "1",
                "--config",
                "/install/spellbook/inference/service--spellbook_streaming_inference.yaml",
            ]
            return (
                await self.create_model_bundle_use_case.execute(
                    user,
                    CreateModelBundleV2Request(
                        name=endpoint_unique_name,
                        schema_location="TBA",
                        flavor=StreamingEnhancedRunnableImageFlavor(
                            flavor=ModelBundleFlavorType.STREAMING_ENHANCED_RUNNABLE_IMAGE,
                            repository="instant-llm",  # TODO: let user choose repo
                            tag=framework_image_tag,
                            command=command,
                            streaming_command=command,
                            env={
                                "MODEL_NAME": model_name,
                            },
                            protocol="http",
                            readiness_initial_delay_seconds=60,
                        ),
                        metadata={},
                    ),
                    do_auth_check=False,
                )
            ).model_bundle_id
        else:
            return (
                await self.create_model_bundle_use_case.execute(
                    user,
                    CreateModelBundleV2Request(
                        name=endpoint_unique_name,
                        schema_location="TBA",
                        flavor=RunnableImageFlavor(
                            flavor=ModelBundleFlavorType.RUNNABLE_IMAGE,
                            repository="instant-llm",
                            tag=framework_image_tag,
                            command=[
                                "dumb-init",
                                "--",
                                "ddtrace-run",
                                "run-service",
                                "--http",
                                "production_threads",
                                "--concurrency",
                                "1",
                                "--config",
                                "/install/spellbook/inference/service--spellbook_inference.yaml",
                            ],
                            env={
                                "MODEL_NAME": model_name,
                            },
                            protocol="http",
                            readiness_initial_delay_seconds=1800,
                        ),
                        metadata={},
                    ),
                    do_auth_check=False,
                )
            ).model_bundle_id

    def _create_vllm_bundle_command(
        self,
        model_name: str,
        framework_image_tag: str,
        num_shards: int,
        quantize: Optional[Quantization],
        checkpoint_path: Optional[str],
        chat_template_override: Optional[str],
        multinode: bool,
        is_worker: bool,
        nodes_per_worker: int = 1,  # only used if multinode
        additional_args: Optional[VLLMEndpointAdditionalArgs] = None,
    ):
        """
        VLLM start command for the single worker, or the leader in a LeaderWorkerSet.
        """
        subcommands = []

        checkpoint_path = get_checkpoint_path(model_name, checkpoint_path)

        # merge additional_args with inferred_additional_args
        # We assume user provided additional args takes precedence over inferred args
        vllm_args = VLLMEndpointAdditionalArgs.model_validate(
            {
                **(
                    infer_addition_engine_args_from_model_name(model_name).model_dump(
                        exclude_none=True
                    )
                ),
                **(additional_args.model_dump(exclude_none=True) if additional_args else {}),
            }
        )

        # added as workaround since transformers doesn't support mistral yet, vllm expects "mistral" in model weights folder
        final_weights_folder = "mistral_files" if "mistral" in model_name else "model_files"
        subcommands += self.load_model_weights_sub_commands(
            LLMInferenceFramework.VLLM,
            framework_image_tag,
            checkpoint_path,
            final_weights_folder,
            trust_remote_code=vllm_args.trust_remote_code or False,
        )

        if multinode:
            if not is_worker:
                ray_cmd = "/workspace/init_ray.sh leader --ray_cluster_size=$RAY_CLUSTER_SIZE --own_address=$K8S_OWN_POD_NAME.$K8S_LWS_NAME.$K8S_OWN_NAMESPACE.svc.cluster.local"
            else:
                ray_cmd = "/workspace/init_ray.sh worker --ray_address=$LWS_LEADER_ADDRESS.svc.cluster.local --own_address=$K8S_OWN_POD_NAME.$K8S_LWS_NAME.$K8S_OWN_NAMESPACE.svc.cluster.local"
            subcommands.append(ray_cmd)

        if not is_worker:
            vllm_args.tensor_parallel_size = num_shards

            if vllm_args.gpu_memory_utilization is not None:
                vllm_args.enforce_eager = True

            if multinode:
                vllm_args.pipeline_parallel_size = nodes_per_worker

            if chat_template_override:
                vllm_args.chat_template = chat_template_override

            if quantize:
                if quantize != Quantization.AWQ:
                    raise InvalidRequestException(
                        f"Quantization {quantize} is not supported by vLLM."
                    )

                vllm_args.quantization = quantize

            if hmi_config.sensitive_log_mode:
                vllm_args.disable_log_requests = True

            vllm_cmd = f"python -m vllm_server --model {final_weights_folder} --served-model-name {model_name} {final_weights_folder} --port 5005"
            for field in VLLMEndpointAdditionalArgs.model_fields.keys():
                config_value = getattr(vllm_args, field, None)
                if config_value is not None:
                    # Special handling for chat_template
                    # Need to encode the chat template as base64 to avoid issues with special characters
                    if field == "chat_template":
                        chat_template_cmd = f'export CHAT_TEMPLATE=$(echo "{encode_template(config_value)}" | base64 --decode)'
                        subcommands.append(chat_template_cmd)
                        config_value = '"$CHAT_TEMPLATE"'

                    # if type of config_value is True, then only need to add the key
                    if isinstance(config_value, bool):
                        if config_value:
                            vllm_cmd += f" --{field.replace('_', '-')}"
                    else:
                        vllm_cmd += f" --{field.replace('_', '-')} {config_value}"

            subcommands.append(vllm_cmd)

        command = [
            "/bin/bash",
            "-c",
            ";".join(subcommands),
        ]

        return command

    async def create_vllm_bundle(
        self,
        user: User,
        model_name: str,
        framework_image_tag: str,
        endpoint_unique_name: str,
        num_shards: int,
        quantize: Optional[Quantization],
        checkpoint_path: Optional[str],
        chat_template_override: Optional[str],
        additional_args: Optional[VLLMEndpointAdditionalArgs] = None,
    ):
        command = self._create_vllm_bundle_command(
            model_name,
            framework_image_tag,
            num_shards,
            quantize,
            checkpoint_path,
            chat_template_override,
            multinode=False,
            is_worker=False,
            nodes_per_worker=1,
            additional_args=additional_args,
        )

        create_model_bundle_v2_request = CreateModelBundleV2Request(
            name=endpoint_unique_name,
            schema_location="TBA",
            flavor=StreamingEnhancedRunnableImageFlavor(
                flavor=ModelBundleFlavorType.STREAMING_ENHANCED_RUNNABLE_IMAGE,
                repository=hmi_config.vllm_repository,
                tag=framework_image_tag,
                command=command,
                streaming_command=command,
                protocol="http",
                readiness_initial_delay_seconds=10,
                healthcheck_route="/health",
                predict_route="/predict",
                streaming_predict_route="/stream",
                extra_routes=[
                    OPENAI_CHAT_COMPLETION_PATH,
                    OPENAI_COMPLETION_PATH,
                ],
                env={},
            ),
            metadata={},
        )

        return (
            await self.create_model_bundle_use_case.execute(
                user,
                create_model_bundle_v2_request,
                do_auth_check=False,
                # Skip auth check because llm create endpoint is called as the user itself,
                # but the user isn't directly making the action. It should come from the fine tune
                # job.
            )
        ).model_bundle_id

    async def create_vllm_multinode_bundle(
        self,
        user: User,
        model_name: str,
        framework_image_tag: str,
        endpoint_unique_name: str,
        num_shards: int,
        nodes_per_worker: int,
        quantize: Optional[Quantization],
        checkpoint_path: Optional[str],
        chat_template_override: Optional[str],
        additional_args: Optional[VLLMEndpointAdditionalArgs] = None,
    ):
        leader_command = self._create_vllm_bundle_command(
            model_name,
            framework_image_tag,
            num_shards,
            quantize,
            checkpoint_path,
            chat_template_override,
            multinode=True,
            is_worker=False,
            nodes_per_worker=nodes_per_worker,
            additional_args=additional_args,
        )
        worker_command = self._create_vllm_bundle_command(
            model_name,
            framework_image_tag,
            num_shards,
            quantize,
            checkpoint_path,
            chat_template_override,
            multinode=True,
            is_worker=True,
            nodes_per_worker=nodes_per_worker,
        )

        # These env vars e.g. K8S_OWN_POD_NAME, K8S_OWN_POD_NAME, K8S_OWN_NAMESPACE, K8S_LWS_CLUSTER_SIZE will be filled in automatically for all LWS pods through
        # Launch's k8s_endpoint_resource_delegate
        common_vllm_envs = {
            "VLLM_HOST_IP": "$(K8S_OWN_POD_NAME).$(K8S_LWS_NAME).$(K8S_OWN_NAMESPACE).svc.cluster.local",  # this needs to match what's given as --own-address in the vllm start command
            "NCCL_SOCKET_IFNAME": "eth0",
            "GLOO_SOCKET_IFNAME": "eth0",  # maybe don't need
            "NCCL_DEBUG": "INFO",  # TODO remove once fully tested, will keep around for now
            "VLLM_LOGGING_LEVEL": "INFO",  # TODO remove once fully tested, will keep around for now
            "RAY_CLUSTER_SIZE": "$(K8S_LWS_CLUSTER_SIZE)",
        }

        create_model_bundle_v2_request = CreateModelBundleV2Request(
            name=endpoint_unique_name,
            schema_location="TBA",
            flavor=StreamingEnhancedRunnableImageFlavor(
                flavor=ModelBundleFlavorType.STREAMING_ENHANCED_RUNNABLE_IMAGE,
                repository=hmi_config.vllm_repository,
                tag=framework_image_tag,
                command=leader_command,
                streaming_command=leader_command,
                protocol="http",
                readiness_initial_delay_seconds=10,
                healthcheck_route="/health",
                predict_route="/predict",
                streaming_predict_route="/stream",
                extra_routes=[OPENAI_CHAT_COMPLETION_PATH, OPENAI_COMPLETION_PATH],
                env=common_vllm_envs,
                worker_command=worker_command,
                worker_env=common_vllm_envs,
            ),
            metadata={},
        )

        return (
            await self.create_model_bundle_use_case.execute(
                user,
                create_model_bundle_v2_request,
                do_auth_check=False,
                # Skip auth check because llm create endpoint is called as the user itself,
                # but the user isn't directly making the action. It should come from the fine tune
                # job.
            )
        ).model_bundle_id

    async def create_lightllm_bundle(
        self,
        user: User,
        model_name: str,
        framework_image_tag: str,
        endpoint_unique_name: str,
        num_shards: int,
        checkpoint_path: Optional[str],
    ):
        command = []

        # TODO: incorporate auto calculate max_total_token_num from https://github.com/ModelTC/lightllm/pull/81
        max_total_token_num = 6000  # LightLLM default
        if num_shards == 1:
            max_total_token_num = 15000  # Default for Llama 2 7B on 1 x A10
        elif num_shards == 2:
            max_total_token_num = 21000  # Default for Llama 2 13B on 2 x A10
        elif num_shards == 4:
            max_total_token_num = 70000  # Default for Llama 2 13B on 4 x A10
        max_req_input_len = 2047
        max_req_total_len = 2048
        if "llama-2" in model_name:
            max_req_input_len = 4095
            max_req_total_len = 4096

        subcommands = []

        checkpoint_path = get_checkpoint_path(model_name, checkpoint_path)
        final_weights_folder = "model_files"
        subcommands += self.load_model_weights_sub_commands(
            LLMInferenceFramework.LIGHTLLM,
            framework_image_tag,
            checkpoint_path,
            final_weights_folder,
        )

        subcommands.append(
            f"python -m lightllm.server.api_server --model_dir {final_weights_folder} --port 5005 --tp {num_shards} --max_total_token_num {max_total_token_num} --max_req_input_len {max_req_input_len} --max_req_total_len {max_req_total_len} --tokenizer_mode auto"
        )

        command = [
            "/bin/bash",
            "-c",
            ";".join(subcommands),
        ]

        return (
            await self.create_model_bundle_use_case.execute(
                user,
                CreateModelBundleV2Request(
                    name=endpoint_unique_name,
                    schema_location="TBA",
                    flavor=StreamingEnhancedRunnableImageFlavor(
                        flavor=ModelBundleFlavorType.STREAMING_ENHANCED_RUNNABLE_IMAGE,
                        repository=hmi_config.lightllm_repository,
                        tag=framework_image_tag,
                        command=command,
                        streaming_command=command,
                        protocol="http",
                        readiness_initial_delay_seconds=10,
                        healthcheck_route="/health",
                        predict_route="/generate",
                        streaming_predict_route="/generate_stream",
                        env={},
                    ),
                    metadata={},
                ),
                do_auth_check=False,
                # Skip auth check because llm create endpoint is called as the user itself,
                # but the user isn't directly making the action. It should come from the fine tune
                # job.
            )
        ).model_bundle_id

    async def create_tensorrt_llm_bundle(
        self,
        user: User,
        framework_image_tag: str,
        endpoint_unique_name: str,
        num_shards: int,
        checkpoint_path: Optional[str],
    ):
        command = []

        subcommands = []

        if not checkpoint_path:
            raise ObjectHasInvalidValueException(
                "Checkpoint must be provided for TensorRT-LLM models."
            )

        validate_checkpoint_path_uri(checkpoint_path)

        subcommands += self.load_model_files_sub_commands_trt_llm(
            checkpoint_path,
        )

        subcommands.append(
            f"python3 launch_triton_server.py --world_size={num_shards} --model_repo=./model_repo/"
        )

        command = [
            "/bin/bash",
            "-c",
            ";".join(subcommands),
        ]

        return (
            await self.create_model_bundle_use_case.execute(
                user,
                CreateModelBundleV2Request(
                    name=endpoint_unique_name,
                    schema_location="TBA",
                    flavor=StreamingEnhancedRunnableImageFlavor(
                        flavor=ModelBundleFlavorType.STREAMING_ENHANCED_RUNNABLE_IMAGE,
                        repository=hmi_config.tensorrt_llm_repository,
                        tag=framework_image_tag,
                        command=command,
                        streaming_command=command,
                        protocol="http",
                        readiness_initial_delay_seconds=10,
                        healthcheck_route="/v2/health/ready",
                        # See https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md
                        predict_route="/v2/models/ensemble/generate",
                        streaming_predict_route="/v2/models/ensemble/generate_stream",
                        env={},
                    ),
                    metadata={},
                ),
                do_auth_check=False,
                # Skip auth check because llm create endpoint is called as the user itself,
                # but the user isn't directly making the action. It should come from the fine tune
                # job.
            )
        ).model_bundle_id


class CreateLLMModelEndpointV1UseCase:
    def __init__(
        self,
        create_llm_model_bundle_use_case: CreateLLMModelBundleV1UseCase,
        model_endpoint_service: ModelEndpointService,
        docker_repository: DockerRepository,
        llm_artifact_gateway: LLMArtifactGateway,
    ):
        self.authz_module = LiveAuthorizationModule()
        self.create_llm_model_bundle_use_case = create_llm_model_bundle_use_case
        self.model_endpoint_service = model_endpoint_service
        self.docker_repository = docker_repository
        self.llm_artifact_gateway = llm_artifact_gateway

    async def execute(
        self, user: User, request: CreateLLMModelEndpointV1Request
    ) -> CreateLLMModelEndpointV1Response:
        await _fill_hardware_info(self.llm_artifact_gateway, request)
        if not (
            request.gpus
            and request.gpu_type
            and request.cpus
            and request.memory
            and request.storage
            and request.nodes_per_worker
        ):
            raise RuntimeError("Some hardware info is missing unexpectedly.")
        validate_deployment_resources(
            min_workers=request.min_workers,
            max_workers=request.max_workers,
            endpoint_type=request.endpoint_type,
            can_scale_http_endpoint_from_zero=self.model_endpoint_service.can_scale_http_endpoint_from_zero(),
        )
        if request.gpu_type == GpuType.NVIDIA_AMPERE_A100E:  # pragma: no cover
            raise ObjectHasInvalidValueException(
                "We have migrated A100 usage to H100. Please request for H100 instead!"
            )
        if request.labels is None:
            raise EndpointLabelsException("Endpoint labels cannot be None!")

        validate_labels(request.labels)
        validate_billing_tags(request.billing_tags)
        validate_post_inference_hooks(user, request.post_inference_hooks)
        validate_model_name(request.model_name, request.inference_framework)
        validate_num_shards(request.num_shards, request.inference_framework, request.gpus)
        validate_quantization(request.quantize, request.inference_framework)
        validate_chat_template(request.chat_template_override, request.inference_framework)

        if request.inference_framework in [
            LLMInferenceFramework.TEXT_GENERATION_INFERENCE,
            LLMInferenceFramework.VLLM,
            LLMInferenceFramework.LIGHTLLM,
            LLMInferenceFramework.TENSORRT_LLM,
        ]:
            if request.endpoint_type != ModelEndpointType.STREAMING:
                raise ObjectHasInvalidValueException(
                    f"Creating endpoint type {str(request.endpoint_type)} is not allowed. Can only create streaming endpoints for text-generation-inference, vLLM, LightLLM, and TensorRT-LLM."
                )

        if request.inference_framework_image_tag == "latest":
            request.inference_framework_image_tag = await _get_latest_tag(
                request.inference_framework
            )

        if (
            request.nodes_per_worker > 1
            and not request.inference_framework == LLMInferenceFramework.VLLM
        ):
            raise ObjectHasInvalidValueException(
                "Multinode endpoints are only supported for VLLM models."
            )

        bundle = await self.create_llm_model_bundle_use_case.execute(
            user,
            endpoint_name=request.name,
            model_name=request.model_name,
            source=request.source,
            framework=request.inference_framework,
            framework_image_tag=request.inference_framework_image_tag,
            endpoint_type=request.endpoint_type,
            num_shards=request.num_shards,
            quantize=request.quantize,
            checkpoint_path=request.checkpoint_path,
            chat_template_override=request.chat_template_override,
            nodes_per_worker=request.nodes_per_worker,
            additional_args=request.model_dump(exclude_none=True),
        )
        validate_resource_requests(
            bundle=bundle,
            cpus=request.cpus,
            memory=request.memory,
            storage=request.storage,
            gpus=request.gpus,
            gpu_type=request.gpu_type,
        )

        prewarm = request.prewarm
        if prewarm is None:
            prewarm = True

        high_priority = request.high_priority
        if high_priority is None:
            high_priority = False

        aws_role = self.authz_module.get_aws_role_for_user(user)
        results_s3_bucket = self.authz_module.get_s3_bucket_for_user(user)

        request.metadata[LLM_METADATA_KEY] = asdict(
            LLMMetadata(
                model_name=request.model_name,
                source=request.source,
                inference_framework=request.inference_framework,
                inference_framework_image_tag=request.inference_framework_image_tag,
                num_shards=request.num_shards,
                quantize=request.quantize,
                checkpoint_path=request.checkpoint_path,
                chat_template_override=request.chat_template_override,
            )
        )

        model_endpoint_record = await self.model_endpoint_service.create_model_endpoint(
            name=request.name,
            created_by=user.user_id,
            model_bundle_id=bundle.id,
            endpoint_type=request.endpoint_type,
            metadata=request.metadata,
            post_inference_hooks=request.post_inference_hooks,
            child_fn_info=None,
            cpus=request.cpus,
            gpus=request.gpus,
            memory=request.memory,
            gpu_type=request.gpu_type,
            storage=request.storage,
            nodes_per_worker=request.nodes_per_worker,
            optimize_costs=bool(request.optimize_costs),
            min_workers=request.min_workers,
            max_workers=request.max_workers,
            per_worker=request.per_worker,
            concurrent_requests_per_worker=LLM_MAX_CONCURRENCY_PER_WORKER,
            labels=request.labels,
            aws_role=aws_role,
            results_s3_bucket=results_s3_bucket,
            prewarm=prewarm,
            high_priority=high_priority,
            owner=user.team_id,
            default_callback_url=request.default_callback_url,
            default_callback_auth=request.default_callback_auth,
            public_inference=request.public_inference,
        )
        _handle_post_inference_hooks(
            created_by=user.user_id,
            name=request.name,
            post_inference_hooks=request.post_inference_hooks,
        )

        await self.model_endpoint_service.get_inference_autoscaling_metrics_gateway().emit_prewarm_metric(
            model_endpoint_record.id
        )

        return CreateLLMModelEndpointV1Response(
            endpoint_creation_task_id=model_endpoint_record.creation_task_id  # type: ignore
        )


class ListLLMModelEndpointsV1UseCase:
    """
    Use case for listing all LLM Model Endpoint of a given user and model endpoint name.
    Also include public_inference LLM endpoints.
    """

    def __init__(self, llm_model_endpoint_service: LLMModelEndpointService):
        self.llm_model_endpoint_service = llm_model_endpoint_service

    async def execute(
        self, user: User, name: Optional[str], order_by: Optional[ModelEndpointOrderBy]
    ) -> ListLLMModelEndpointsV1Response:
        """
        Runs the use case to list all Model Endpoints owned by the user with the given name.

        Args:
            user: The owner of the model endpoint(s).
            name: The name of the Model Endpoint(s).
            order_by: An optional argument to specify the output ordering of the model endpoints.

        Returns:
            A response object that contains the model endpoints.
        """
        model_endpoints = await self.llm_model_endpoint_service.list_llm_model_endpoints(
            owner=user.team_id, name=name, order_by=order_by
        )
        return ListLLMModelEndpointsV1Response(
            model_endpoints=[
                _model_endpoint_entity_to_get_llm_model_endpoint_response(m)
                for m in model_endpoints
            ]
        )


class GetLLMModelEndpointByNameV1UseCase:
    """
    Use case for getting an LLM Model Endpoint of a given user by name.
    """

    def __init__(self, llm_model_endpoint_service: LLMModelEndpointService):
        self.llm_model_endpoint_service = llm_model_endpoint_service
        self.authz_module = LiveAuthorizationModule()

    async def execute(self, user: User, model_endpoint_name: str) -> GetLLMModelEndpointV1Response:
        """
        Runs the use case to get the LLM endpoint with the given name.

        Args:
            user: The owner of the model endpoint.
            model_endpoint_name: The name of the model endpoint.

        Returns:
            A response object that contains the model endpoint.

        Raises:
            ObjectNotFoundException: If a model endpoint with the given name could not be found.
            ObjectNotAuthorizedException: If the owner does not own the model endpoint.
        """
        model_endpoint = await self.llm_model_endpoint_service.get_llm_model_endpoint(
            model_endpoint_name
        )
        if not model_endpoint:
            raise ObjectNotFoundException
        if not self.authz_module.check_access_read_owned_entity(
            user, model_endpoint.record
        ) and not self.authz_module.check_endpoint_public_inference_for_user(
            user, model_endpoint.record
        ):
            raise ObjectNotAuthorizedException
        return _model_endpoint_entity_to_get_llm_model_endpoint_response(model_endpoint)


def merge_metadata(
    request: Optional[Dict[str, Any]], record: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if request is None:
        return record
    if record is None:
        return request
    return {**record, **request}


class UpdateLLMModelEndpointV1UseCase:
    def __init__(
        self,
        create_llm_model_bundle_use_case: CreateLLMModelBundleV1UseCase,
        model_endpoint_service: ModelEndpointService,
        llm_model_endpoint_service: LLMModelEndpointService,
        docker_repository: DockerRepository,
    ):
        self.authz_module = LiveAuthorizationModule()
        self.create_llm_model_bundle_use_case = create_llm_model_bundle_use_case
        self.model_endpoint_service = model_endpoint_service
        self.llm_model_endpoint_service = llm_model_endpoint_service
        self.docker_repository = docker_repository

    async def execute(
        self,
        user: User,
        model_endpoint_name: str,
        request: UpdateLLMModelEndpointV1Request,
    ) -> UpdateLLMModelEndpointV1Response:
        if request.labels is not None:
            validate_labels(request.labels)
        validate_billing_tags(request.billing_tags)
        validate_post_inference_hooks(user, request.post_inference_hooks)

        model_endpoint = await self.llm_model_endpoint_service.get_llm_model_endpoint(
            model_endpoint_name
        )
        if not model_endpoint:
            raise ObjectNotFoundException
        if not self.authz_module.check_access_write_owned_entity(user, model_endpoint.record):
            raise ObjectNotAuthorizedException

        endpoint_record = model_endpoint.record
        model_endpoint_id = endpoint_record.id
        bundle = endpoint_record.current_model_bundle

        # TODO: We may want to consider what happens if an endpoint gets stuck in UPDATE_PENDING
        #  on first creating it, and we need to find a way to get it unstuck. This would end up
        # causing endpoint.infra_state to be None.
        if model_endpoint.infra_state is None:
            error_msg = f"Endpoint infra state not found for {model_endpoint_name=}"
            logger.error(error_msg)
            raise EndpointInfraStateNotFound(error_msg)

        infra_state = model_endpoint.infra_state
        metadata: Optional[Dict[str, Any]]

        if (
            request.force_bundle_recreation
            or request.model_name
            or request.source
            or request.inference_framework_image_tag
            or request.num_shards
            or request.quantize
            or request.checkpoint_path
            or request.chat_template_override
        ):
            llm_metadata = (model_endpoint.record.metadata or {}).get(LLM_METADATA_KEY, {})
            inference_framework = llm_metadata["inference_framework"]

            if request.inference_framework_image_tag == "latest":
                inference_framework_image_tag = await _get_latest_tag(inference_framework)
            else:
                inference_framework_image_tag = (
                    request.inference_framework_image_tag
                    or llm_metadata["inference_framework_image_tag"]
                )

            model_name = request.model_name or llm_metadata["model_name"]
            source = request.source or llm_metadata["source"]
            num_shards = request.num_shards or llm_metadata["num_shards"]
            quantize = request.quantize or llm_metadata.get("quantize")
            checkpoint_path = request.checkpoint_path or llm_metadata.get("checkpoint_path")

            validate_model_name(model_name, inference_framework)
            validate_num_shards(
                num_shards,
                inference_framework,
                request.gpus or infra_state.resource_state.gpus,
            )
            validate_quantization(quantize, inference_framework)
            validate_chat_template(request.chat_template_override, inference_framework)
            chat_template_override = request.chat_template_override or llm_metadata.get(
                "chat_template_override"
            )

            bundle = await self.create_llm_model_bundle_use_case.execute(
                user,
                endpoint_name=model_endpoint_name,
                model_name=model_name,
                source=source,
                framework=inference_framework,
                framework_image_tag=inference_framework_image_tag,
                endpoint_type=endpoint_record.endpoint_type,
                num_shards=num_shards,
                quantize=quantize,
                checkpoint_path=checkpoint_path,
                chat_template_override=chat_template_override,
                nodes_per_worker=model_endpoint.infra_state.resource_state.nodes_per_worker,
                additional_args=request.model_dump(exclude_none=True),
            )

            metadata = endpoint_record.metadata or {}
            metadata[LLM_METADATA_KEY] = asdict(
                LLMMetadata(
                    model_name=model_name,
                    source=source,
                    inference_framework=inference_framework,
                    inference_framework_image_tag=inference_framework_image_tag,
                    num_shards=num_shards,
                    quantize=quantize,
                    checkpoint_path=checkpoint_path,
                    chat_template_override=chat_template_override,
                )
            )
            endpoint_record.metadata = metadata

        # For resources that are not specified in the update endpoint request, pass in resource from
        # infra_state to make sure that after the update, all resources are valid and in sync.
        # E.g. If user only want to update gpus and leave gpu_type as None, we use the existing gpu_type
        # from infra_state to avoid passing in None to validate_resource_requests.
        validate_resource_requests(
            bundle=bundle,
            cpus=request.cpus or infra_state.resource_state.cpus,
            memory=request.memory or infra_state.resource_state.memory,
            storage=request.storage or infra_state.resource_state.storage,
            gpus=request.gpus or infra_state.resource_state.gpus,
            gpu_type=request.gpu_type or infra_state.resource_state.gpu_type,
        )

        validate_deployment_resources(
            min_workers=request.min_workers,
            max_workers=request.max_workers,
            endpoint_type=endpoint_record.endpoint_type,
            can_scale_http_endpoint_from_zero=self.model_endpoint_service.can_scale_http_endpoint_from_zero(),
        )

        if request.metadata is not None:
            # If reserved metadata key is provided, throw ObjectHasInvalidValueException
            for key in RESERVED_METADATA_KEYS:
                if key in request.metadata:
                    raise ObjectHasInvalidValueException(
                        f"{key} is a reserved metadata key and cannot be used by user."
                    )

        metadata = merge_metadata(request.metadata, endpoint_record.metadata)

        updated_endpoint_record = await self.model_endpoint_service.update_model_endpoint(
            model_endpoint_id=model_endpoint_id,
            model_bundle_id=bundle.id,
            metadata=metadata,
            post_inference_hooks=request.post_inference_hooks,
            cpus=request.cpus,
            gpus=request.gpus,
            memory=request.memory,
            gpu_type=request.gpu_type,
            storage=request.storage,
            optimize_costs=request.optimize_costs,
            min_workers=request.min_workers,
            max_workers=request.max_workers,
            per_worker=request.per_worker,
            concurrent_requests_per_worker=None,  # Don't need to update this value presumably
            labels=request.labels,
            prewarm=request.prewarm,
            high_priority=request.high_priority,
            default_callback_url=request.default_callback_url,
            default_callback_auth=request.default_callback_auth,
            public_inference=request.public_inference,
        )
        _handle_post_inference_hooks(
            created_by=endpoint_record.created_by,
            name=updated_endpoint_record.name,
            post_inference_hooks=request.post_inference_hooks,
        )

        return UpdateLLMModelEndpointV1Response(
            endpoint_creation_task_id=updated_endpoint_record.creation_task_id  # type: ignore
        )


class DeleteLLMEndpointByNameUseCase:
    """
    Use case for deleting an LLM Model Endpoint of a given user by endpoint name.
    """

    def __init__(
        self,
        model_endpoint_service: ModelEndpointService,
        llm_model_endpoint_service: LLMModelEndpointService,
    ):
        self.model_endpoint_service = model_endpoint_service
        self.llm_model_endpoint_service = llm_model_endpoint_service
        self.authz_module = LiveAuthorizationModule()

    async def execute(self, user: User, model_endpoint_name: str) -> DeleteLLMEndpointResponse:
        """
        Runs the use case to delete the LLM endpoint owned by the user with the given name.

        Args:
            user: The owner of the model endpoint.
            model_endpoint_name: The name of the model endpoint.

        Returns:
            A response object that contains a boolean indicating if deletion was successful.

        Raises:
            ObjectNotFoundException: If a model endpoint with the given name could not be found.
            ObjectNotAuthorizedException: If the owner does not own the model endpoint.
        """
        model_endpoints = await self.llm_model_endpoint_service.list_llm_model_endpoints(
            owner=user.user_id, name=model_endpoint_name, order_by=None
        )
        if len(model_endpoints) != 1:
            raise ObjectNotFoundException
        model_endpoint = model_endpoints[0]
        if not self.authz_module.check_access_write_owned_entity(user, model_endpoint.record):
            raise ObjectNotAuthorizedException
        await self.model_endpoint_service.delete_model_endpoint(model_endpoint.record.id)
        return DeleteLLMEndpointResponse(deleted=True)


def deepspeed_result_to_tokens(result: Dict[str, Any]) -> List[TokenOutput]:
    tokens = []
    for i in range(len(result["token_probs"]["token_probs"])):
        tokens.append(
            TokenOutput(
                token=result["token_probs"]["tokens"][i],
                log_prob=math.log(result["token_probs"]["token_probs"][i]),
            )
        )
    return tokens


def validate_and_update_completion_params(
    inference_framework: LLMInferenceFramework,
    request: Union[CompletionSyncV1Request, CompletionStreamV1Request],
) -> Union[CompletionSyncV1Request, CompletionStreamV1Request]:
    # top_k, top_p
    if inference_framework in [
        LLMInferenceFramework.TEXT_GENERATION_INFERENCE,
        LLMInferenceFramework.VLLM,
        LLMInferenceFramework.LIGHTLLM,
    ]:
        if request.temperature == 0:
            if request.top_k not in [-1, None] or request.top_p not in [1.0, None]:
                raise ObjectHasInvalidValueException(
                    "top_k and top_p can't be enabled when temperature is 0."
                )
        if request.top_k == 0:
            raise ObjectHasInvalidValueException(
                "top_k needs to be strictly positive, or set it to be -1 / None to disable top_k."
            )
        if inference_framework == LLMInferenceFramework.TEXT_GENERATION_INFERENCE:
            request.top_k = None if request.top_k == -1 else request.top_k
            request.top_p = None if request.top_p == 1.0 else request.top_p
        if inference_framework in [
            LLMInferenceFramework.VLLM,
            LLMInferenceFramework.LIGHTLLM,
        ]:
            request.top_k = -1 if request.top_k is None else request.top_k
            request.top_p = 1.0 if request.top_p is None else request.top_p
    else:
        if request.top_k or request.top_p:
            raise ObjectHasInvalidValueException(
                "top_k and top_p are only supported in text-generation-inference, vllm, lightllm."
            )

    # presence_penalty, frequency_penalty
    if inference_framework in [
        LLMInferenceFramework.VLLM,
        LLMInferenceFramework.LIGHTLLM,
    ]:
        request.presence_penalty = (
            0.0 if request.presence_penalty is None else request.presence_penalty
        )
        request.frequency_penalty = (
            0.0 if request.frequency_penalty is None else request.frequency_penalty
        )
    else:
        if request.presence_penalty or request.frequency_penalty:
            raise ObjectHasInvalidValueException(
                "presence_penalty and frequency_penalty are only supported in vllm, lightllm."
            )

    # return_token_log_probs
    if inference_framework in [
        LLMInferenceFramework.DEEPSPEED,
        LLMInferenceFramework.TEXT_GENERATION_INFERENCE,
        LLMInferenceFramework.VLLM,
        LLMInferenceFramework.LIGHTLLM,
    ]:
        pass
    else:
        if request.return_token_log_probs:
            raise ObjectHasInvalidValueException(
                "return_token_log_probs is only supported in deepspeed, text-generation-inference, vllm, lightllm."
            )

    # include_stop_str_in_output
    if inference_framework == LLMInferenceFramework.VLLM:
        pass
    else:
        if request.include_stop_str_in_output is not None:
            raise ObjectHasInvalidValueException(
                "include_stop_str_in_output is only supported in vllm."
            )

    guided_count = 0
    if request.guided_choice is not None:
        guided_count += 1
    if request.guided_json is not None:
        guided_count += 1
    if request.guided_regex is not None:
        guided_count += 1
    if request.guided_grammar is not None:
        guided_count += 1

    if guided_count > 1:
        raise ObjectHasInvalidValueException(
            "Only one of guided_json, guided_choice, guided_regex, guided_grammar can be enabled."
        )

    if (
        request.guided_choice is not None
        or request.guided_regex is not None
        or request.guided_json is not None
        or request.guided_grammar is not None
    ) and not inference_framework == LLMInferenceFramework.VLLM:
        raise ObjectHasInvalidValueException("Guided decoding is only supported in vllm.")

    return request


class CompletionSyncV1UseCase:
    """
    Use case for running a prompt completion on an LLM endpoint.
    """

    def __init__(
        self,
        model_endpoint_service: ModelEndpointService,
        llm_model_endpoint_service: LLMModelEndpointService,
        tokenizer_repository: TokenizerRepository,
    ):
        self.model_endpoint_service = model_endpoint_service
        self.llm_model_endpoint_service = llm_model_endpoint_service
        self.authz_module = LiveAuthorizationModule()
        self.tokenizer_repository = tokenizer_repository

    def model_output_to_completion_output(
        self,
        model_output: Dict[str, Any],
        model_endpoint: ModelEndpoint,
        prompt: str,
        with_token_probs: Optional[bool],
    ) -> CompletionOutput:
        model_content = _model_endpoint_entity_to_get_llm_model_endpoint_response(model_endpoint)
        if model_content.inference_framework == LLMInferenceFramework.DEEPSPEED:
            completion_token_count = len(model_output["token_probs"]["tokens"])
            tokens = None
            if with_token_probs:
                tokens = deepspeed_result_to_tokens(model_output)
            return CompletionOutput(
                text=model_output["text"],
                num_prompt_tokens=count_tokens(
                    prompt,
                    model_content.model_name,
                    self.tokenizer_repository,
                ),
                num_completion_tokens=completion_token_count,
                tokens=tokens,
            )
        elif model_content.inference_framework == LLMInferenceFramework.TEXT_GENERATION_INFERENCE:
            try:
                tokens = None
                if with_token_probs:
                    tokens = [
                        TokenOutput(token=t["text"], log_prob=t["logprob"])
                        for t in model_output["details"]["tokens"]
                    ]
                return CompletionOutput(
                    text=model_output["generated_text"],
                    num_prompt_tokens=len(model_output["details"]["prefill"]),
                    num_completion_tokens=model_output["details"]["generated_tokens"],
                    tokens=tokens,
                )
            except Exception:
                logger.exception(f"Error parsing text-generation-inference output {model_output}.")
                if model_output.get("error_type") == "validation":
                    raise InvalidRequestException(model_output.get("error"))  # trigger a 400
                else:
                    raise UpstreamServiceError(
                        status_code=500, content=bytes(model_output["error"], "utf-8")
                    )

        elif model_content.inference_framework == LLMInferenceFramework.VLLM:
            tokens = None
            if with_token_probs:
                tokens = [
                    TokenOutput(
                        token=model_output["tokens"][index],
                        log_prob=list(t.values())[0],
                    )
                    for index, t in enumerate(model_output["log_probs"])
                ]
            return CompletionOutput(
                text=model_output["text"],
                num_prompt_tokens=model_output["count_prompt_tokens"],
                num_completion_tokens=model_output["count_output_tokens"],
                tokens=tokens,
            )
        elif model_content.inference_framework == LLMInferenceFramework.LIGHTLLM:
            tokens = None
            if with_token_probs:
                tokens = [
                    TokenOutput(token=t["text"], log_prob=t["logprob"])
                    for t in model_output["tokens"]
                ]
            return CompletionOutput(
                text=model_output["generated_text"][0],
                num_prompt_tokens=count_tokens(
                    prompt,
                    model_content.model_name,
                    self.tokenizer_repository,
                ),
                num_completion_tokens=model_output["count_output_tokens"],
                tokens=tokens,
            )
        elif model_content.inference_framework == LLMInferenceFramework.TENSORRT_LLM:
            if not model_content.model_name:
                raise InvalidRequestException(
                    f"Invalid endpoint {model_content.name} has no base model"
                )
            if not prompt:
                raise InvalidRequestException("Prompt must be provided for TensorRT-LLM models.")
            num_prompt_tokens = count_tokens(
                prompt, model_content.model_name, self.tokenizer_repository
            )
            if "token_ids" in model_output:
                # TensorRT 23.10 has this field, TensorRT 24.03 does not
                # For backwards compatibility with pre-2024/05/02
                num_completion_tokens = len(model_output["token_ids"]) - num_prompt_tokens
                # Output is "<s> prompt output"
                text = model_output["text_output"][(len(prompt) + 4) :]
            elif "output_log_probs" in model_output:
                # TensorRT 24.01 + surrounding code.
                # For some reason TRT returns output_log_probs as either a list or a float
                # Also the log probs don't look right, so returning log-probs is still broken
                num_completion_tokens = (
                    len(model_output["output_log_probs"])
                    if type(model_output["output_log_probs"]) is list
                    else 1
                )
                # Output is just "output". See `exclude_input_in_output` inside of
                # inference/tensorrt-llm/triton_model_repo/tensorrt_llm/config.pbtxt
                text = model_output["text_output"]
            return CompletionOutput(
                text=text,
                num_prompt_tokens=num_prompt_tokens,
                num_completion_tokens=num_completion_tokens,
            )
        else:
            raise EndpointUnsupportedInferenceTypeException(
                f"Unsupported inference framework {model_content.inference_framework}"
            )

    async def execute(
        self, user: User, model_endpoint_name: str, request: CompletionSyncV1Request
    ) -> CompletionSyncV1Response:
        """
        Runs the use case to create a sync inference task.

        Args:
            user: The user who is creating the sync inference task.
            model_endpoint_name: The name of the model endpoint for the task.
            request: The body of the request to forward to the endpoint.

        Returns:
            A response object that contains the status and result of the task.

        Raises:
            ObjectNotFoundException: If a model endpoint with the given name could not be found.
            ObjectNotAuthorizedException: If the owner does not own the model endpoint.
        """

        request_id = LoggerTagManager.get(LoggerTagKey.REQUEST_ID)
        add_trace_request_id(request_id)

        model_endpoints = await self.llm_model_endpoint_service.list_llm_model_endpoints(
            owner=user.team_id, name=model_endpoint_name, order_by=None
        )

        if len(model_endpoints) == 0:
            raise ObjectNotFoundException

        if len(model_endpoints) > 1:
            raise ObjectHasInvalidValueException(
                f"Expected 1 LLM model endpoint for model name {model_endpoint_name}, got {len(model_endpoints)}"
            )

        add_trace_model_name(model_endpoint_name)

        model_endpoint = model_endpoints[0]

        if not self.authz_module.check_access_read_owned_entity(
            user, model_endpoint.record
        ) and not self.authz_module.check_endpoint_public_inference_for_user(
            user, model_endpoint.record
        ):
            raise ObjectNotAuthorizedException

        if model_endpoint.record.endpoint_type not in [
            ModelEndpointType.SYNC,
            ModelEndpointType.STREAMING,
        ]:
            raise EndpointUnsupportedInferenceTypeException(
                f"Endpoint {model_endpoint_name} does not serve sync requests."
            )

        inference_gateway = self.model_endpoint_service.get_sync_model_endpoint_inference_gateway()
        autoscaling_metrics_gateway = (
            self.model_endpoint_service.get_inference_autoscaling_metrics_gateway()
        )
        await autoscaling_metrics_gateway.emit_inference_autoscaling_metric(
            endpoint_id=model_endpoint.record.id
        )
        endpoint_content = _model_endpoint_entity_to_get_llm_model_endpoint_response(model_endpoint)

        manually_resolve_dns = (
            model_endpoint.infra_state is not None
            and model_endpoint.infra_state.resource_state.nodes_per_worker > 1
            and hmi_config.istio_enabled
        )
        validated_request = validate_and_update_completion_params(
            endpoint_content.inference_framework, request
        )
        if not isinstance(validated_request, CompletionSyncV1Request):
            raise ValueError(
                f"request has type {validated_request.__class__.__name__}, expected type CompletionSyncV1Request"
            )
        request = validated_request

        if endpoint_content.inference_framework == LLMInferenceFramework.DEEPSPEED:
            args: Any = {
                "prompts": [request.prompt],
                "token_probs": True,
                "generate_kwargs": {
                    "do_sample": True,
                    "temperature": request.temperature,
                    "max_new_tokens": request.max_new_tokens,
                },
                "serialize_results_as_string": False,
            }
            if request.stop_sequences is not None:
                # Deepspeed models only accepts one stop sequence
                args["stop_sequence"] = request.stop_sequences[0]

            inference_request = SyncEndpointPredictV1Request(
                args=args,
                num_retries=NUM_DOWNSTREAM_REQUEST_RETRIES,
                timeout_seconds=DOWNSTREAM_REQUEST_TIMEOUT_SECONDS,
            )
            predict_result = await inference_gateway.predict(
                topic=model_endpoint.record.destination,
                predict_request=inference_request,
                manually_resolve_dns=manually_resolve_dns,
                endpoint_name=model_endpoint.record.name,
            )

            if predict_result.status == TaskStatus.SUCCESS and predict_result.result is not None:
                return CompletionSyncV1Response(
                    request_id=request_id,
                    output=self.model_output_to_completion_output(
                        predict_result.result["result"][0],
                        model_endpoint,
                        request.prompt,
                        request.return_token_log_probs,
                    ),
                )
            else:
                raise UpstreamServiceError(
                    status_code=500,
                    content=(
                        predict_result.traceback.encode("utf-8")
                        if predict_result.traceback is not None
                        else b""
                    ),
                )
        elif (
            endpoint_content.inference_framework == LLMInferenceFramework.TEXT_GENERATION_INFERENCE
        ):
            tgi_args: Any = {
                "inputs": request.prompt,
                "parameters": {
                    "max_new_tokens": request.max_new_tokens,
                    "decoder_input_details": True,
                },
            }
            if request.stop_sequences is not None:
                tgi_args["parameters"]["stop"] = request.stop_sequences
            if request.temperature > 0:
                tgi_args["parameters"]["temperature"] = request.temperature
                tgi_args["parameters"]["do_sample"] = True
                tgi_args["parameters"]["top_k"] = request.top_k
                tgi_args["parameters"]["top_p"] = request.top_p
            else:
                tgi_args["parameters"]["do_sample"] = False

            inference_request = SyncEndpointPredictV1Request(
                args=tgi_args,
                num_retries=NUM_DOWNSTREAM_REQUEST_RETRIES,
                timeout_seconds=DOWNSTREAM_REQUEST_TIMEOUT_SECONDS,
            )
            predict_result = await inference_gateway.predict(
                topic=model_endpoint.record.destination,
                predict_request=inference_request,
                manually_resolve_dns=manually_resolve_dns,
                endpoint_name=model_endpoint.record.name,
            )

            if predict_result.status != TaskStatus.SUCCESS or predict_result.result is None:
                raise UpstreamServiceError(
                    status_code=500,
                    content=(
                        predict_result.traceback.encode("utf-8")
                        if predict_result.traceback is not None
                        else b""
                    ),
                )

            output = json.loads(predict_result.result["result"])

            return CompletionSyncV1Response(
                request_id=request_id,
                output=self.model_output_to_completion_output(
                    output,
                    model_endpoint,
                    request.prompt,
                    request.return_token_log_probs,
                ),
            )
        elif endpoint_content.inference_framework == LLMInferenceFramework.VLLM:
            vllm_args: Any = {
                "prompt": request.prompt,
                "max_tokens": request.max_new_tokens,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
            }
            if request.stop_sequences is not None:
                vllm_args["stop"] = request.stop_sequences
            vllm_args["temperature"] = request.temperature
            if request.temperature > 0:
                vllm_args["top_k"] = request.top_k
                vllm_args["top_p"] = request.top_p
            if request.return_token_log_probs:
                vllm_args["logprobs"] = 1
            if request.include_stop_str_in_output is not None:
                vllm_args["include_stop_str_in_output"] = request.include_stop_str_in_output
            if request.guided_choice is not None:
                vllm_args["guided_choice"] = request.guided_choice
            if request.guided_regex is not None:
                vllm_args["guided_regex"] = request.guided_regex
            if request.guided_json is not None:
                vllm_args["guided_json"] = request.guided_json
            if request.guided_grammar is not None:
                vllm_args["guided_grammar"] = request.guided_grammar
            if request.skip_special_tokens is not None:
                vllm_args["skip_special_tokens"] = request.skip_special_tokens

            inference_request = SyncEndpointPredictV1Request(
                args=vllm_args,
                num_retries=NUM_DOWNSTREAM_REQUEST_RETRIES,
                timeout_seconds=DOWNSTREAM_REQUEST_TIMEOUT_SECONDS,
            )
            predict_result = await inference_gateway.predict(
                topic=model_endpoint.record.destination,
                predict_request=inference_request,
                manually_resolve_dns=manually_resolve_dns,
                endpoint_name=model_endpoint.record.name,
            )

            if predict_result.status != TaskStatus.SUCCESS or predict_result.result is None:
                raise UpstreamServiceError(
                    status_code=500,
                    content=(
                        predict_result.traceback.encode("utf-8")
                        if predict_result.traceback is not None
                        else b""
                    ),
                )

            output = json.loads(predict_result.result["result"])
            return CompletionSyncV1Response(
                request_id=request_id,
                output=self.model_output_to_completion_output(
                    output,
                    model_endpoint,
                    request.prompt,
                    request.return_token_log_probs,
                ),
            )
        elif endpoint_content.inference_framework == LLMInferenceFramework.LIGHTLLM:
            lightllm_args: Any = {
                "inputs": request.prompt,
                "parameters": {
                    "max_new_tokens": request.max_new_tokens,
                    "presence_penalty": request.presence_penalty,
                    "frequency_penalty": request.frequency_penalty,
                },
            }
            # TODO: implement stop sequences
            if request.temperature > 0:
                lightllm_args["parameters"]["temperature"] = request.temperature
                lightllm_args["parameters"]["do_sample"] = True
                lightllm_args["top_k"] = request.top_k
                lightllm_args["top_p"] = request.top_p
            else:
                lightllm_args["parameters"]["do_sample"] = False
            if request.return_token_log_probs:
                lightllm_args["parameters"]["return_details"] = True

            inference_request = SyncEndpointPredictV1Request(
                args=lightllm_args,
                num_retries=NUM_DOWNSTREAM_REQUEST_RETRIES,
                timeout_seconds=DOWNSTREAM_REQUEST_TIMEOUT_SECONDS,
            )
            predict_result = await inference_gateway.predict(
                topic=model_endpoint.record.destination,
                predict_request=inference_request,
                manually_resolve_dns=manually_resolve_dns,
                endpoint_name=model_endpoint.record.name,
            )

            if predict_result.status != TaskStatus.SUCCESS or predict_result.result is None:
                raise UpstreamServiceError(
                    status_code=500,
                    content=(
                        predict_result.traceback.encode("utf-8")
                        if predict_result.traceback is not None
                        else b""
                    ),
                )

            output = json.loads(predict_result.result["result"])
            return CompletionSyncV1Response(
                request_id=request_id,
                output=self.model_output_to_completion_output(
                    output,
                    model_endpoint,
                    request.prompt,
                    request.return_token_log_probs,
                ),
            )
        elif endpoint_content.inference_framework == LLMInferenceFramework.TENSORRT_LLM:
            # TODO: Stop sequences is buggy and return token logprobs are not supported
            # TODO: verify the implementation of presence_penalty and repetition_penalty
            # and see if they fit our existing definition of presence_penalty and frequency_penalty
            # Ref https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/sampling_penalty_kernels.cu
            trt_llm_args: Any = {
                "text_input": request.prompt,
                "max_tokens": request.max_new_tokens,
                "stop_words": request.stop_sequences if request.stop_sequences else "",
                "bad_words": "",
                "temperature": request.temperature,
            }

            inference_request = SyncEndpointPredictV1Request(
                args=trt_llm_args,
                num_retries=NUM_DOWNSTREAM_REQUEST_RETRIES,
                timeout_seconds=DOWNSTREAM_REQUEST_TIMEOUT_SECONDS,
            )
            predict_result = await inference_gateway.predict(
                topic=model_endpoint.record.destination,
                predict_request=inference_request,
                manually_resolve_dns=manually_resolve_dns,
                endpoint_name=model_endpoint.record.name,
            )

            if predict_result.status != TaskStatus.SUCCESS or predict_result.result is None:
                raise UpstreamServiceError(
                    status_code=500,
                    content=(
                        predict_result.traceback.encode("utf-8")
                        if predict_result.traceback is not None
                        else b""
                    ),
                )

            output = json.loads(predict_result.result["result"])
            return CompletionSyncV1Response(
                request_id=request_id,
                output=self.model_output_to_completion_output(
                    output,
                    model_endpoint,
                    request.prompt,
                    request.return_token_log_probs,
                ),
            )
        else:
            raise EndpointUnsupportedInferenceTypeException(
                f"Unsupported inference framework {endpoint_content.inference_framework}"
            )


class CompletionStreamV1UseCase:
    """
    Use case for running a stream prompt completion on an LLM endpoint.
    """

    def __init__(
        self,
        model_endpoint_service: ModelEndpointService,
        llm_model_endpoint_service: LLMModelEndpointService,
        tokenizer_repository: TokenizerRepository,
    ):
        self.model_endpoint_service = model_endpoint_service
        self.llm_model_endpoint_service = llm_model_endpoint_service
        self.authz_module = LiveAuthorizationModule()
        self.tokenizer_repository = tokenizer_repository

    async def execute(
        self, user: User, model_endpoint_name: str, request: CompletionStreamV1Request
    ) -> AsyncIterable[CompletionStreamV1Response]:
        """
        Runs the use case to create a stream inference task.
        NOTE: Must be called with await(), since the function is not a generator itself, but rather creates one and
        returns a reference to it. This structure allows exceptions that occur before response streaming begins
        to propagate to the client as HTTP exceptions with the appropriate code.

        Args:
            user: The user who is creating the stream inference task.
            model_endpoint_name: The name of the model endpoint for the task.
            request: The body of the request to forward to the endpoint.

        Returns:
            An asynchronous response chunk generator, containing response objects to be iterated through with 'async for'.
            Each response object contains the status and result of the task.

        Raises:
            ObjectNotFoundException: If a model endpoint with the given name could not be found.
            ObjectHasInvalidValueException: If there are multiple model endpoints with the given name.
            ObjectNotAuthorizedException: If the owner does not own the model endpoint.
            EndpointUnsupportedInferenceTypeException: If the model endpoint does not support streaming or uses
                an unsupported inference framework.
            UpstreamServiceError: If an error occurs upstream in the streaming inference API call.
            InvalidRequestException: If request validation fails during inference.
        """

        request_id = LoggerTagManager.get(LoggerTagKey.REQUEST_ID)
        add_trace_request_id(request_id)

        model_endpoints = await self.llm_model_endpoint_service.list_llm_model_endpoints(
            owner=user.team_id, name=model_endpoint_name, order_by=None
        )

        if len(model_endpoints) == 0:
            raise ObjectNotFoundException(f"Model endpoint {model_endpoint_name} not found.")

        if len(model_endpoints) > 1:
            raise ObjectHasInvalidValueException(
                f"Expected 1 LLM model endpoint for model name {model_endpoint_name}, got {len(model_endpoints)}"
            )

        add_trace_model_name(model_endpoint_name)

        model_endpoint = model_endpoints[0]

        if not self.authz_module.check_access_read_owned_entity(
            user, model_endpoint.record
        ) and not self.authz_module.check_endpoint_public_inference_for_user(
            user, model_endpoint.record
        ):
            raise ObjectNotAuthorizedException

        if model_endpoint.record.endpoint_type != ModelEndpointType.STREAMING:
            raise EndpointUnsupportedInferenceTypeException(
                f"Endpoint {model_endpoint_name} is not a streaming endpoint."
            )

        inference_gateway = (
            self.model_endpoint_service.get_streaming_model_endpoint_inference_gateway()
        )
        autoscaling_metrics_gateway = (
            self.model_endpoint_service.get_inference_autoscaling_metrics_gateway()
        )
        await autoscaling_metrics_gateway.emit_inference_autoscaling_metric(
            endpoint_id=model_endpoint.record.id
        )

        model_content = _model_endpoint_entity_to_get_llm_model_endpoint_response(model_endpoint)
        validated_request = validate_and_update_completion_params(
            model_content.inference_framework, request
        )
        if not isinstance(validated_request, CompletionStreamV1Request):
            raise ValueError(
                f"request has type {validated_request.__class__.__name__}, expected type CompletionStreamV1Request"
            )
        request = validated_request

        manually_resolve_dns = (
            model_endpoint.infra_state is not None
            and model_endpoint.infra_state.resource_state.nodes_per_worker > 1
            and hmi_config.istio_enabled
        )

        args: Any = None
        num_prompt_tokens = None
        if model_content.inference_framework == LLMInferenceFramework.DEEPSPEED:
            args = {
                "prompts": [request.prompt],
                "token_probs": True,
                "generate_kwargs": {
                    "do_sample": True,
                    "temperature": request.temperature,
                    "max_new_tokens": request.max_new_tokens,
                },
                "serialize_results_as_string": False,
            }
            if request.stop_sequences is not None:
                # Deepspeed models only accepts one stop sequence
                args["stop_sequence"] = request.stop_sequences[0]
            num_prompt_tokens = count_tokens(
                request.prompt,
                model_content.model_name,
                self.tokenizer_repository,
            )
        elif model_content.inference_framework == LLMInferenceFramework.TEXT_GENERATION_INFERENCE:
            args = {
                "inputs": request.prompt,
                "parameters": {
                    "max_new_tokens": request.max_new_tokens,
                },
            }
            if request.stop_sequences is not None:
                args["parameters"]["stop"] = request.stop_sequences
            if request.temperature > 0:
                args["parameters"]["temperature"] = request.temperature
                args["parameters"]["do_sample"] = True
                args["parameters"]["top_k"] = request.top_k
                args["parameters"]["top_p"] = request.top_p
            else:
                args["parameters"]["do_sample"] = False
            num_prompt_tokens = count_tokens(
                request.prompt,
                model_content.model_name,
                self.tokenizer_repository,
            )
        elif model_content.inference_framework == LLMInferenceFramework.VLLM:
            args = {
                "prompt": request.prompt,
                "max_tokens": request.max_new_tokens,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
            }
            if request.stop_sequences is not None:
                args["stop"] = request.stop_sequences
            args["temperature"] = request.temperature
            if request.temperature > 0:
                args["top_k"] = request.top_k
                args["top_p"] = request.top_p
            if request.return_token_log_probs:
                args["logprobs"] = 1
            if request.include_stop_str_in_output is not None:
                args["include_stop_str_in_output"] = request.include_stop_str_in_output
            if request.guided_choice is not None:
                args["guided_choice"] = request.guided_choice
            if request.guided_regex is not None:
                args["guided_regex"] = request.guided_regex
            if request.guided_json is not None:
                args["guided_json"] = request.guided_json
            if request.guided_grammar is not None:
                args["guided_grammar"] = request.guided_grammar
            if request.skip_special_tokens is not None:
                args["skip_special_tokens"] = request.skip_special_tokens
            args["stream"] = True
        elif model_content.inference_framework == LLMInferenceFramework.LIGHTLLM:
            args = {
                "inputs": request.prompt,
                "parameters": {
                    "max_new_tokens": request.max_new_tokens,
                    "presence_penalty": request.presence_penalty,
                    "frequency_penalty": request.frequency_penalty,
                },
            }
            # TODO: stop sequences
            if request.temperature > 0:
                args["parameters"]["temperature"] = request.temperature
                args["parameters"]["do_sample"] = True
                args["parameters"]["top_k"] = request.top_k
                args["parameters"]["top_p"] = request.top_p
            else:
                args["parameters"]["do_sample"] = False
            if request.return_token_log_probs:
                args["parameters"]["return_details"] = True
            num_prompt_tokens = count_tokens(
                request.prompt,
                model_content.model_name,
                self.tokenizer_repository,
            )
        elif model_content.inference_framework == LLMInferenceFramework.TENSORRT_LLM:
            # TODO: Stop sequences is buggy and return token logprobs are not supported
            # TODO: verify the implementation of presence_penalty and repetition_penalty
            # and see if they fit our existing definition of presence_penalty and frequency_penalty
            # Ref https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/sampling_penalty_kernels.cu
            args = {
                "text_input": request.prompt,
                "max_tokens": request.max_new_tokens,
                "stop_words": request.stop_sequences if request.stop_sequences else "",
                "bad_words": "",
                "temperature": request.temperature,
                "stream": True,
            }
            num_prompt_tokens = count_tokens(
                request.prompt,
                model_content.model_name,
                self.tokenizer_repository,
            )
        else:
            raise EndpointUnsupportedInferenceTypeException(
                f"Unsupported inference framework {model_content.inference_framework}"
            )

        inference_request = SyncEndpointPredictV1Request(
            args=args,
            num_retries=NUM_DOWNSTREAM_REQUEST_RETRIES,
            timeout_seconds=DOWNSTREAM_REQUEST_TIMEOUT_SECONDS,
        )

        return self._response_chunk_generator(
            request=request,
            request_id=request_id,
            model_endpoint=model_endpoint,
            model_content=model_content,
            inference_gateway=inference_gateway,
            inference_request=inference_request,
            num_prompt_tokens=num_prompt_tokens,
            manually_resolve_dns=manually_resolve_dns,
        )

    async def _response_chunk_generator(
        self,
        request: CompletionStreamV1Request,
        request_id: Optional[str],
        model_endpoint: ModelEndpoint,
        model_content: GetLLMModelEndpointV1Response,
        inference_gateway: StreamingModelEndpointInferenceGateway,
        inference_request: SyncEndpointPredictV1Request,
        num_prompt_tokens: Optional[int],
        manually_resolve_dns: bool,
    ) -> AsyncIterable[CompletionStreamV1Response]:
        """
        Async generator yielding tokens to stream for the completions response. Should only be called when
        returned directly by execute().
        """
        predict_result = inference_gateway.streaming_predict(
            topic=model_endpoint.record.destination,
            predict_request=inference_request,
            manually_resolve_dns=manually_resolve_dns,
            endpoint_name=model_endpoint.record.name,
        )

        num_completion_tokens = 0
        async for res in predict_result:
            if not res.status == TaskStatus.SUCCESS or res.result is None:
                # Raise an UpstreamServiceError if the task has failed
                if res.status == TaskStatus.FAILURE:
                    raise UpstreamServiceError(
                        status_code=500,
                        content=(
                            res.traceback.encode("utf-8") if res.traceback is not None else b""
                        ),
                    )
                # Otherwise, yield empty response chunk for unsuccessful or empty results
                yield CompletionStreamV1Response(
                    request_id=request_id,
                    output=None,
                )
            else:
                result = res.result
                # DEEPSPEED
                if model_content.inference_framework == LLMInferenceFramework.DEEPSPEED:
                    if "token" in result["result"]:
                        yield CompletionStreamV1Response(
                            request_id=request_id,
                            output=CompletionStreamOutput(
                                text=result["result"]["token"],
                                finished=False,
                                num_prompt_tokens=None,
                                num_completion_tokens=None,
                            ),
                        )
                    else:
                        completion_token_count = len(
                            result["result"]["response"][0]["token_probs"]["tokens"]
                        )
                        yield CompletionStreamV1Response(
                            request_id=request_id,
                            output=CompletionStreamOutput(
                                text=result["result"]["response"][0]["text"],
                                finished=True,
                                num_prompt_tokens=num_prompt_tokens,
                                num_completion_tokens=completion_token_count,
                            ),
                        )
                # TEXT_GENERATION_INTERFACE
                elif (
                    model_content.inference_framework
                    == LLMInferenceFramework.TEXT_GENERATION_INFERENCE
                ):
                    if result["result"].get("generated_text") is not None:
                        finished = True
                    else:
                        finished = False

                    num_completion_tokens += 1

                    token = None
                    if request.return_token_log_probs:
                        token = TokenOutput(
                            token=result["result"]["token"]["text"],
                            log_prob=result["result"]["token"]["logprob"],
                        )
                    try:
                        yield CompletionStreamV1Response(
                            request_id=request_id,
                            output=CompletionStreamOutput(
                                text=result["result"]["token"]["text"],
                                finished=finished,
                                num_prompt_tokens=(num_prompt_tokens if finished else None),
                                num_completion_tokens=num_completion_tokens,
                                token=token,
                            ),
                        )
                    except Exception:
                        logger.exception(
                            f"Error parsing text-generation-inference output. Result: {result['result']}"
                        )
                        if result["result"].get("error_type") == "validation":
                            raise InvalidRequestException(
                                result["result"].get("error")
                            )  # trigger a 400
                        else:
                            raise UpstreamServiceError(
                                status_code=500, content=result.get("error")
                            )  # also change llms_v1.py that will return a 500 HTTPException so user can retry
                # VLLM
                elif model_content.inference_framework == LLMInferenceFramework.VLLM:
                    token = None
                    if request.return_token_log_probs:
                        token = TokenOutput(
                            token=result["result"]["text"],
                            log_prob=list(result["result"]["log_probs"].values())[0],
                        )
                    finished = result["result"]["finished"]
                    num_prompt_tokens = result["result"]["count_prompt_tokens"]
                    yield CompletionStreamV1Response(
                        request_id=request_id,
                        output=CompletionStreamOutput(
                            text=result["result"]["text"],
                            finished=finished,
                            num_prompt_tokens=num_prompt_tokens if finished else None,
                            num_completion_tokens=result["result"]["count_output_tokens"],
                            token=token,
                        ),
                    )
                # LIGHTLLM
                elif model_content.inference_framework == LLMInferenceFramework.LIGHTLLM:
                    token = None
                    num_completion_tokens += 1
                    if request.return_token_log_probs:
                        token = TokenOutput(
                            token=result["result"]["token"]["text"],
                            log_prob=result["result"]["token"]["logprob"],
                        )
                    finished = result["result"]["finished"]
                    yield CompletionStreamV1Response(
                        request_id=request_id,
                        output=CompletionStreamOutput(
                            text=result["result"]["token"]["text"],
                            finished=finished,
                            num_prompt_tokens=num_prompt_tokens if finished else None,
                            num_completion_tokens=num_completion_tokens,
                            token=token,
                        ),
                    )
                # TENSORRT_LLM
                elif model_content.inference_framework == LLMInferenceFramework.TENSORRT_LLM:
                    num_completion_tokens += 1
                    yield CompletionStreamV1Response(
                        request_id=request_id,
                        output=CompletionStreamOutput(
                            text=result["result"]["text_output"],
                            finished=False,  # Tracked by https://github.com/NVIDIA/TensorRT-LLM/issues/240
                            num_prompt_tokens=num_prompt_tokens,
                            num_completion_tokens=num_completion_tokens,
                        ),
                    )
                # No else clause needed for an unsupported inference framework, since we check
                # model_content.inference_framework in execute() prior to calling _response_chunk_generator,
                # raising an exception if it is not one of the frameworks handled above.


def validate_endpoint_supports_openai_completion(
    endpoint: ModelEndpoint, endpoint_content: GetLLMModelEndpointV1Response
):  # pragma: no cover
    if endpoint_content.inference_framework not in OPENAI_SUPPORTED_INFERENCE_FRAMEWORKS:
        raise EndpointUnsupportedInferenceTypeException(
            f"The endpoint's inference framework ({endpoint_content.inference_framework}) does not support openai compatible completion."
        )

    if (
        not isinstance(endpoint.record.current_model_bundle.flavor, RunnableImageLike)
        or OPENAI_COMPLETION_PATH not in endpoint.record.current_model_bundle.flavor.extra_routes
    ):
        raise EndpointUnsupportedRequestException(
            "Endpoint does not support v2 openai compatible completion"
        )


class CompletionSyncV2UseCase:
    """
    Use case for running a v2 openai compatible completion on an LLM endpoint.
    """

    def __init__(
        self,
        model_endpoint_service: ModelEndpointService,
        llm_model_endpoint_service: LLMModelEndpointService,
        tokenizer_repository: TokenizerRepository,
    ):  # pragma: no cover
        self.model_endpoint_service = model_endpoint_service
        self.llm_model_endpoint_service = llm_model_endpoint_service
        self.authz_module = LiveAuthorizationModule()
        self.tokenizer_repository = tokenizer_repository

    async def execute(
        self, user: User, model_endpoint_name: str, request: CompletionV2Request
    ) -> CompletionV2SyncResponse:  # pragma: no cover
        """
        Runs the use case to create a sync inference task.

        Args:
            user: The user who is creating the sync inference task.
            model_endpoint_name: The name of the model endpoint for the task.
            request: The body of the request to forward to the endpoint.

        Returns:
            A response object that contains the status and result of the task.

        Raises:
            ObjectNotFoundException: If a model endpoint with the given name could not be found.
            ObjectNotAuthorizedException: If the owner does not own the model endpoint.
        """

        request_id = LoggerTagManager.get(LoggerTagKey.REQUEST_ID)
        add_trace_request_id(request_id)

        model_endpoints = await self.llm_model_endpoint_service.list_llm_model_endpoints(
            owner=user.team_id, name=model_endpoint_name, order_by=None
        )

        if len(model_endpoints) == 0:
            raise ObjectNotFoundException

        if len(model_endpoints) > 1:
            raise ObjectHasInvalidValueException(
                f"Expected 1 LLM model endpoint for model name {model_endpoint_name}, got {len(model_endpoints)}"
            )

        add_trace_model_name(model_endpoint_name)

        model_endpoint = model_endpoints[0]

        if not self.authz_module.check_access_read_owned_entity(
            user, model_endpoint.record
        ) and not self.authz_module.check_endpoint_public_inference_for_user(
            user, model_endpoint.record
        ):
            raise ObjectNotAuthorizedException

        if (
            model_endpoint.record.endpoint_type is not ModelEndpointType.SYNC
            and model_endpoint.record.endpoint_type is not ModelEndpointType.STREAMING
        ):
            raise EndpointUnsupportedInferenceTypeException(
                f"Endpoint {model_endpoint_name} does not serve sync requests."
            )

        inference_gateway = self.model_endpoint_service.get_sync_model_endpoint_inference_gateway()
        autoscaling_metrics_gateway = (
            self.model_endpoint_service.get_inference_autoscaling_metrics_gateway()
        )
        await autoscaling_metrics_gateway.emit_inference_autoscaling_metric(
            endpoint_id=model_endpoint.record.id
        )
        endpoint_content = _model_endpoint_entity_to_get_llm_model_endpoint_response(model_endpoint)

        manually_resolve_dns = (
            model_endpoint.infra_state is not None
            and model_endpoint.infra_state.resource_state.nodes_per_worker > 1
            and hmi_config.istio_enabled
        )

        validate_endpoint_supports_openai_completion(model_endpoint, endpoint_content)

        # if inference framework is VLLM, we need to set the model to use the weights folder
        if endpoint_content.inference_framework == LLMInferenceFramework.VLLM:
            request.model = VLLM_MODEL_WEIGHTS_FOLDER

        inference_request = SyncEndpointPredictV1Request(
            args=request.model_dump(exclude_none=True),
            destination_path=OPENAI_COMPLETION_PATH,
            num_retries=NUM_DOWNSTREAM_REQUEST_RETRIES,
            timeout_seconds=DOWNSTREAM_REQUEST_TIMEOUT_SECONDS,
        )
        try:
            predict_result = await inference_gateway.predict(
                topic=model_endpoint.record.destination,
                predict_request=inference_request,
                manually_resolve_dns=manually_resolve_dns,
                endpoint_name=model_endpoint.record.name,
            )

            if predict_result.status != TaskStatus.SUCCESS or predict_result.result is None:
                raise UpstreamServiceError(
                    status_code=500,
                    content=(
                        predict_result.traceback.encode("utf-8")
                        if predict_result.traceback is not None
                        else b""
                    ),
                )

            output = json.loads(predict_result.result["result"])
            # reset model name to correct value
            output["model"] = model_endpoint.record.name
            return CompletionV2SyncResponse.model_validate(output)
        except UpstreamServiceError as exc:
            # Expect upstream inference service to handle bulk of input validation
            if 400 <= exc.status_code < 500:
                raise InvalidRequestException(exc.content)
            raise exc


class CompletionStreamV2UseCase:
    """
    Use case for running a v2 openai compatible completion on an LLM endpoint.
    """

    def __init__(
        self,
        model_endpoint_service: ModelEndpointService,
        llm_model_endpoint_service: LLMModelEndpointService,
        tokenizer_repository: TokenizerRepository,
    ):  # pragma: no cover
        self.model_endpoint_service = model_endpoint_service
        self.llm_model_endpoint_service = llm_model_endpoint_service
        self.authz_module = LiveAuthorizationModule()
        self.tokenizer_repository = tokenizer_repository

    async def execute(
        self, model_endpoint_name: str, request: CompletionV2Request, user: User
    ) -> AsyncGenerator[CompletionV2StreamSuccessChunk, None]:  # pragma: no cover
        request_id = LoggerTagManager.get(LoggerTagKey.REQUEST_ID)
        add_trace_request_id(request_id)

        model_endpoints = await self.llm_model_endpoint_service.list_llm_model_endpoints(
            owner=user.team_id, name=model_endpoint_name, order_by=None
        )

        if len(model_endpoints) == 0:
            raise ObjectNotFoundException(f"Model endpoint {model_endpoint_name} not found.")

        if len(model_endpoints) > 1:
            raise ObjectHasInvalidValueException(
                f"Expected 1 LLM model endpoint for model name {model_endpoint_name}, got {len(model_endpoints)}"
            )

        add_trace_model_name(model_endpoint_name)

        model_endpoint = model_endpoints[0]

        if not self.authz_module.check_access_read_owned_entity(
            user, model_endpoint.record
        ) and not self.authz_module.check_endpoint_public_inference_for_user(
            user, model_endpoint.record
        ):
            raise ObjectNotAuthorizedException

        if model_endpoint.record.endpoint_type != ModelEndpointType.STREAMING:
            raise EndpointUnsupportedInferenceTypeException(
                f"Endpoint {model_endpoint_name} is not a streaming endpoint."
            )

        inference_gateway = (
            self.model_endpoint_service.get_streaming_model_endpoint_inference_gateway()
        )
        autoscaling_metrics_gateway = (
            self.model_endpoint_service.get_inference_autoscaling_metrics_gateway()
        )
        await autoscaling_metrics_gateway.emit_inference_autoscaling_metric(
            endpoint_id=model_endpoint.record.id
        )

        model_content = _model_endpoint_entity_to_get_llm_model_endpoint_response(model_endpoint)

        manually_resolve_dns = (
            model_endpoint.infra_state is not None
            and model_endpoint.infra_state.resource_state.nodes_per_worker > 1
            and hmi_config.istio_enabled
        )

        validate_endpoint_supports_openai_completion(model_endpoint, model_content)

        # if inference framework is VLLM, we need to set the model to use the weights folder
        if model_content.inference_framework == LLMInferenceFramework.VLLM:
            request.model = VLLM_MODEL_WEIGHTS_FOLDER

        inference_request = SyncEndpointPredictV1Request(
            args=request.model_dump(exclude_none=True),
            destination_path=OPENAI_COMPLETION_PATH,
            num_retries=NUM_DOWNSTREAM_REQUEST_RETRIES,
            timeout_seconds=DOWNSTREAM_REQUEST_TIMEOUT_SECONDS,
        )

        return self._response_chunk_generator(
            request_id=request_id,
            model_endpoint=model_endpoint,
            model_content=model_content,
            inference_gateway=inference_gateway,
            inference_request=inference_request,
            manually_resolve_dns=manually_resolve_dns,
        )

    async def _response_chunk_generator(
        self,
        request_id: Optional[str],
        model_endpoint: ModelEndpoint,
        model_content: GetLLMModelEndpointV1Response,
        inference_gateway: StreamingModelEndpointInferenceGateway,
        inference_request: SyncEndpointPredictV1Request,
        manually_resolve_dns: bool,
    ) -> AsyncGenerator[CompletionV2StreamSuccessChunk, None]:  # pragma: no cover
        """
        Async generator yielding tokens to stream for the completions response. Should only be called when
        returned directly by execute().
        """
        try:
            predict_result = inference_gateway.streaming_predict(
                topic=model_endpoint.record.destination,
                predict_request=inference_request,
                manually_resolve_dns=manually_resolve_dns,
                endpoint_name=model_endpoint.record.name,
            )
        except UpstreamServiceError as exc:
            # Expect upstream inference service to handle bulk of input validation
            if 400 <= exc.status_code < 500:
                raise InvalidRequestException(str(exc))

            raise exc

        async for res in predict_result:
            if not res.status == TaskStatus.SUCCESS or res.result is None:
                raise UpstreamServiceError(
                    status_code=500,
                    content=(res.traceback.encode("utf-8") if res.traceback is not None else b""),
                )
            else:
                result = res.result["result"]
                # Reset model name to correct value
                if "DONE" in result:
                    continue
                result["model"] = model_endpoint.record.name
                yield CompletionV2StreamSuccessChunk.model_validate(result)


def validate_endpoint_supports_chat_completion(
    endpoint: ModelEndpoint, endpoint_content: GetLLMModelEndpointV1Response
):  # pragma: no cover
    if endpoint_content.inference_framework not in CHAT_SUPPORTED_INFERENCE_FRAMEWORKS:
        raise EndpointUnsupportedInferenceTypeException(
            f"The endpoint's inference framework ({endpoint_content.inference_framework}) does not support chat completion."
        )

    if (
        not isinstance(endpoint.record.current_model_bundle.flavor, RunnableImageLike)
        or OPENAI_CHAT_COMPLETION_PATH
        not in endpoint.record.current_model_bundle.flavor.extra_routes
    ):
        raise EndpointUnsupportedRequestException("Endpoint does not support chat completion")


class ChatCompletionSyncV2UseCase:
    """
    Use case for running a chat completion on an LLM endpoint.
    """

    def __init__(
        self,
        model_endpoint_service: ModelEndpointService,
        llm_model_endpoint_service: LLMModelEndpointService,
        tokenizer_repository: TokenizerRepository,
    ):
        self.model_endpoint_service = model_endpoint_service
        self.llm_model_endpoint_service = llm_model_endpoint_service
        self.authz_module = LiveAuthorizationModule()
        self.tokenizer_repository = tokenizer_repository

    async def execute(
        self, user: User, model_endpoint_name: str, request: ChatCompletionV2Request
    ) -> ChatCompletionV2SyncResponse:  # pragma: no cover
        """
        Runs the use case to create a sync inference task.

        Args:
            user: The user who is creating the sync inference task.
            model_endpoint_name: The name of the model endpoint for the task.
            request: The body of the request to forward to the endpoint.

        Returns:
            A response object that contains the status and result of the task.

        Raises:
            ObjectNotFoundException: If a model endpoint with the given name could not be found.
            ObjectNotAuthorizedException: If the owner does not own the model endpoint.
        """

        request_id = LoggerTagManager.get(LoggerTagKey.REQUEST_ID)
        add_trace_request_id(request_id)

        model_endpoints = await self.llm_model_endpoint_service.list_llm_model_endpoints(
            owner=user.team_id, name=model_endpoint_name, order_by=None
        )

        if len(model_endpoints) == 0:
            raise ObjectNotFoundException

        if len(model_endpoints) > 1:
            raise ObjectHasInvalidValueException(
                f"Expected 1 LLM model endpoint for model name {model_endpoint_name}, got {len(model_endpoints)}"
            )

        add_trace_model_name(model_endpoint_name)

        model_endpoint = model_endpoints[0]

        if not self.authz_module.check_access_read_owned_entity(
            user, model_endpoint.record
        ) and not self.authz_module.check_endpoint_public_inference_for_user(
            user, model_endpoint.record
        ):
            raise ObjectNotAuthorizedException

        if (
            model_endpoint.record.endpoint_type is not ModelEndpointType.SYNC
            and model_endpoint.record.endpoint_type is not ModelEndpointType.STREAMING
        ):
            raise EndpointUnsupportedInferenceTypeException(
                f"Endpoint {model_endpoint_name} does not serve sync requests."
            )

        inference_gateway = self.model_endpoint_service.get_sync_model_endpoint_inference_gateway()
        autoscaling_metrics_gateway = (
            self.model_endpoint_service.get_inference_autoscaling_metrics_gateway()
        )
        await autoscaling_metrics_gateway.emit_inference_autoscaling_metric(
            endpoint_id=model_endpoint.record.id
        )
        endpoint_content = _model_endpoint_entity_to_get_llm_model_endpoint_response(model_endpoint)

        manually_resolve_dns = (
            model_endpoint.infra_state is not None
            and model_endpoint.infra_state.resource_state.nodes_per_worker > 1
            and hmi_config.istio_enabled
        )

        validate_endpoint_supports_chat_completion(model_endpoint, endpoint_content)

        # if inference framework is VLLM, we need to set the model to use the weights folder
        if endpoint_content.inference_framework == LLMInferenceFramework.VLLM:
            request.model = VLLM_MODEL_WEIGHTS_FOLDER

        inference_request = SyncEndpointPredictV1Request(
            args=request.model_dump(exclude_none=True),
            destination_path=OPENAI_CHAT_COMPLETION_PATH,
            num_retries=NUM_DOWNSTREAM_REQUEST_RETRIES,
            timeout_seconds=DOWNSTREAM_REQUEST_TIMEOUT_SECONDS,
        )
        try:
            predict_result = await inference_gateway.predict(
                topic=model_endpoint.record.destination,
                predict_request=inference_request,
                manually_resolve_dns=manually_resolve_dns,
                endpoint_name=model_endpoint.record.name,
            )

            if predict_result.status != TaskStatus.SUCCESS or predict_result.result is None:
                raise UpstreamServiceError(
                    status_code=500,
                    content=(
                        predict_result.traceback.encode("utf-8")
                        if predict_result.traceback is not None
                        else b""
                    ),
                )

            output = json.loads(predict_result.result["result"])
            # reset model name to correct value
            output["model"] = model_endpoint.record.name
            return ChatCompletionV2SyncResponse.model_validate(output)
        except UpstreamServiceError as exc:
            # Expect upstream inference service to handle bulk of input validation
            if 400 <= exc.status_code < 500:
                raise InvalidRequestException(exc.content)
            raise exc


class ChatCompletionStreamV2UseCase:
    """
    Use case for running a chat completion on an LLM endpoint.
    """

    def __init__(
        self,
        model_endpoint_service: ModelEndpointService,
        llm_model_endpoint_service: LLMModelEndpointService,
        tokenizer_repository: TokenizerRepository,
    ):
        self.model_endpoint_service = model_endpoint_service
        self.llm_model_endpoint_service = llm_model_endpoint_service
        self.authz_module = LiveAuthorizationModule()
        self.tokenizer_repository = tokenizer_repository

    async def execute(
        self, model_endpoint_name: str, request: ChatCompletionV2Request, user: User
    ) -> AsyncGenerator[ChatCompletionV2StreamSuccessChunk, None]:  # pragma: no cover
        request_id = LoggerTagManager.get(LoggerTagKey.REQUEST_ID)
        add_trace_request_id(request_id)

        model_endpoints = await self.llm_model_endpoint_service.list_llm_model_endpoints(
            owner=user.team_id, name=model_endpoint_name, order_by=None
        )

        if len(model_endpoints) == 0:
            raise ObjectNotFoundException(f"Model endpoint {model_endpoint_name} not found.")

        if len(model_endpoints) > 1:
            raise ObjectHasInvalidValueException(
                f"Expected 1 LLM model endpoint for model name {model_endpoint_name}, got {len(model_endpoints)}"
            )

        add_trace_model_name(model_endpoint_name)

        model_endpoint = model_endpoints[0]

        if not self.authz_module.check_access_read_owned_entity(
            user, model_endpoint.record
        ) and not self.authz_module.check_endpoint_public_inference_for_user(
            user, model_endpoint.record
        ):
            raise ObjectNotAuthorizedException

        if model_endpoint.record.endpoint_type != ModelEndpointType.STREAMING:
            raise EndpointUnsupportedInferenceTypeException(
                f"Endpoint {model_endpoint_name} is not a streaming endpoint."
            )

        inference_gateway = (
            self.model_endpoint_service.get_streaming_model_endpoint_inference_gateway()
        )
        autoscaling_metrics_gateway = (
            self.model_endpoint_service.get_inference_autoscaling_metrics_gateway()
        )
        await autoscaling_metrics_gateway.emit_inference_autoscaling_metric(
            endpoint_id=model_endpoint.record.id
        )

        model_content = _model_endpoint_entity_to_get_llm_model_endpoint_response(model_endpoint)

        manually_resolve_dns = (
            model_endpoint.infra_state is not None
            and model_endpoint.infra_state.resource_state.nodes_per_worker > 1
            and hmi_config.istio_enabled
        )
        validate_endpoint_supports_chat_completion(model_endpoint, model_content)

        # if inference framework is VLLM, we need to set the model to use the weights folder
        if model_content.inference_framework == LLMInferenceFramework.VLLM:
            request.model = VLLM_MODEL_WEIGHTS_FOLDER

        inference_request = SyncEndpointPredictV1Request(
            args=request.model_dump(exclude_none=True),
            destination_path=OPENAI_CHAT_COMPLETION_PATH,
            num_retries=NUM_DOWNSTREAM_REQUEST_RETRIES,
            timeout_seconds=DOWNSTREAM_REQUEST_TIMEOUT_SECONDS,
        )

        return self._response_chunk_generator(
            request_id=request_id,
            model_endpoint=model_endpoint,
            model_content=model_content,
            inference_gateway=inference_gateway,
            inference_request=inference_request,
            manually_resolve_dns=manually_resolve_dns,
        )

    async def _response_chunk_generator(
        self,
        request_id: Optional[str],
        model_endpoint: ModelEndpoint,
        model_content: GetLLMModelEndpointV1Response,
        inference_gateway: StreamingModelEndpointInferenceGateway,
        inference_request: SyncEndpointPredictV1Request,
        manually_resolve_dns: bool,
    ) -> AsyncGenerator[ChatCompletionV2StreamSuccessChunk, None]:
        """
        Async generator yielding tokens to stream for the completions response. Should only be called when
        returned directly by execute().
        """
        try:
            predict_result = inference_gateway.streaming_predict(
                topic=model_endpoint.record.destination,
                predict_request=inference_request,
                manually_resolve_dns=manually_resolve_dns,
                endpoint_name=model_endpoint.record.name,
            )
        except UpstreamServiceError as exc:
            # Expect upstream inference service to handle bulk of input validation
            if 400 <= exc.status_code < 500:
                raise InvalidRequestException(str(exc))

            raise exc

        async for res in predict_result:
            if not res.status == TaskStatus.SUCCESS or res.result is None:
                raise UpstreamServiceError(
                    status_code=500,
                    content=(res.traceback.encode("utf-8") if res.traceback is not None else b""),
                )
            else:
                result = res.result["result"]
                # Reset model name to correct value
                if "DONE" in result:
                    continue
                result["model"] = model_endpoint.record.name
                yield ChatCompletionV2StreamSuccessChunk.model_validate(result)


class ModelDownloadV1UseCase:
    def __init__(
        self,
        filesystem_gateway: FilesystemGateway,
        model_endpoint_service: ModelEndpointService,
        llm_artifact_gateway: LLMArtifactGateway,
    ):
        self.filesystem_gateway = filesystem_gateway
        self.model_endpoint_service = model_endpoint_service
        self.llm_artifact_gateway = llm_artifact_gateway

    async def execute(self, user: User, request: ModelDownloadRequest) -> ModelDownloadResponse:
        model_endpoints = await self.model_endpoint_service.list_model_endpoints(
            owner=user.team_id, name=request.model_name, order_by=None
        )
        if len(model_endpoints) == 0:
            raise ObjectNotFoundException

        if len(model_endpoints) > 1:
            raise ObjectHasInvalidValueException(
                f"Expected 1 LLM model endpoint for model name {request.model_name}, got {len(model_endpoints)}"
            )
        model_files = self.llm_artifact_gateway.get_model_weights_urls(
            user.team_id, request.model_name
        )
        urls = {}
        for model_file in model_files:
            # don't want to make s3 bucket full keys public, so trim to just keep file name
            public_file_name = model_file.rsplit("/", 1)[-1]
            urls[public_file_name] = self.filesystem_gateway.generate_signed_url(model_file)
        return ModelDownloadResponse(urls=urls)


async def _fill_hardware_info(
    llm_artifact_gateway: LLMArtifactGateway, request: CreateLLMModelEndpointV1Request
):
    if (
        request.gpus is None
        or request.gpu_type is None
        or request.cpus is None
        or request.memory is None
        or request.storage is None
        or request.nodes_per_worker is None
    ):
        if not (
            request.gpus is None
            and request.gpu_type is None
            and request.cpus is None
            and request.memory is None
            and request.storage is None
            and request.nodes_per_worker is None
        ):
            raise ObjectHasInvalidValueException(
                "All hardware spec fields (gpus, gpu_type, cpus, memory, storage, nodes_per_worker) must be provided if any hardware spec field is missing."
            )
        checkpoint_path = get_checkpoint_path(request.model_name, request.checkpoint_path)
        hardware_info = await _infer_hardware(
            llm_artifact_gateway, request.model_name, checkpoint_path
        )
        request.gpus = hardware_info.gpus
        request.gpu_type = hardware_info.gpu_type
        request.cpus = hardware_info.cpus
        request.memory = hardware_info.memory
        request.storage = hardware_info.storage
        request.nodes_per_worker = hardware_info.nodes_per_worker
        if hardware_info.gpus:  # make lint happy
            request.num_shards = hardware_info.gpus


def get_model_param_count_b(model_name: str) -> int:
    """Get the number of parameters in the model in billions"""
    if "mixtral-8x7b" in model_name:
        model_param_count_b = 47
    elif "mixtral-8x22b" in model_name:
        model_param_count_b = 140
    elif "phi-3-mini" in model_name:
        model_param_count_b = 4
    elif "phi-3-small" in model_name:
        model_param_count_b = 8
    elif "phi-3-medium" in model_name:
        model_param_count_b = 15
    elif "deepseek-coder-v2-lite" in model_name:
        model_param_count_b = 16
    elif "deepseek-coder-v2" in model_name:
        model_param_count_b = 237
    else:
        numbers = re.findall(r"(\d+)b", model_name)
        if len(numbers) == 0:
            raise ObjectHasInvalidValueException(
                f"Unable to infer number of parameters for {model_name}."
            )
        model_param_count_b = int(numbers[-1])
    return model_param_count_b


@lru_cache()
async def _infer_hardware(
    llm_artifact_gateway: LLMArtifactGateway,
    model_name: str,
    checkpoint_path: str,
    is_batch_job: bool = False,
    max_context_length: Optional[int] = None,
) -> CreateDockerImageBatchJobResourceRequests:
    config = llm_artifact_gateway.get_model_config(checkpoint_path)

    dtype_size = 2
    kv_multiplier = 20 if is_batch_job else 2

    max_position_embeddings = (
        min(max_context_length, config["max_position_embeddings"])
        if max_context_length
        else config["max_position_embeddings"]
    )

    min_kv_cache_size = (
        kv_multiplier
        * dtype_size
        * config["num_hidden_layers"]
        * config["hidden_size"]
        * max_position_embeddings
        // (config["num_attention_heads"] // config["num_key_value_heads"])
    )

    model_param_count_b = get_model_param_count_b(model_name)
    model_weights_size = dtype_size * model_param_count_b * 1_000_000_000

    min_memory_gb = math.ceil((min_kv_cache_size + model_weights_size) / 1_000_000_000 / 0.9)

    logger.info(
        f"Memory calculation result: {min_memory_gb=} for {model_name} context_size: {max_position_embeddings}, min_kv_cache_size: {min_kv_cache_size}, model_weights_size: {model_weights_size}, is_batch_job: {is_batch_job}"
    )

    config_map = await _get_recommended_hardware_config_map()
    by_model_name = {item["name"]: item for item in yaml.safe_load(config_map["byModelName"])}
    by_gpu_memory_gb = yaml.safe_load(config_map["byGpuMemoryGb"])
    if model_name in by_model_name:
        cpus = by_model_name[model_name]["cpus"]
        gpus = by_model_name[model_name]["gpus"]
        memory = by_model_name[model_name]["memory"]
        storage = by_model_name[model_name]["storage"]
        gpu_type = by_model_name[model_name]["gpu_type"]
        nodes_per_worker = by_model_name[model_name]["nodes_per_worker"]
    else:
        by_gpu_memory_gb = sorted(by_gpu_memory_gb, key=lambda x: x["gpu_memory_le"])
        for recs in by_gpu_memory_gb:
            if min_memory_gb <= recs["gpu_memory_le"]:
                cpus = recs["cpus"]
                gpus = recs["gpus"]
                memory = recs["memory"]
                storage = recs["storage"]
                gpu_type = recs["gpu_type"]
                nodes_per_worker = recs["nodes_per_worker"]
                break
        else:
            raise ObjectHasInvalidValueException(f"Unable to infer hardware for {model_name}.")

    return CreateDockerImageBatchJobResourceRequests(
        cpus=cpus,
        gpus=gpus,
        memory=memory,
        storage=storage,
        gpu_type=gpu_type,
        nodes_per_worker=nodes_per_worker,
    )


def infer_addition_engine_args_from_model_name(
    model_name: str,
) -> VLLMEndpointAdditionalArgs:
    # Increase max gpu utilization for larger models
    gpu_memory_utilization = 0.9
    try:
        model_param_count_b = get_model_param_count_b(model_name)
        if model_param_count_b >= 70:
            gpu_memory_utilization = 0.95
    except ObjectHasInvalidValueException:  # pragma: no cover
        pass

    # Gemma 2 requires flashinfer attention backend
    attention_backend = None
    if model_name.startswith("gemma-2"):
        attention_backend = "FLASHINFER"

    trust_remote_code = None
    # DeepSeek requires trust_remote_code
    if model_name.startswith("deepseek"):
        trust_remote_code = True

    return VLLMEndpointAdditionalArgs(
        gpu_memory_utilization=gpu_memory_utilization,
        attention_backend=attention_backend,
        trust_remote_code=trust_remote_code,
    )


class CreateBatchCompletionsUseCase:
    def __init__(
        self,
        docker_image_batch_job_gateway: DockerImageBatchJobGateway,
        docker_repository: DockerRepository,
        docker_image_batch_job_bundle_repo: DockerImageBatchJobBundleRepository,
        llm_artifact_gateway: LLMArtifactGateway,
    ):
        self.docker_image_batch_job_gateway = docker_image_batch_job_gateway
        self.docker_repository = docker_repository
        self.docker_image_batch_job_bundle_repo = docker_image_batch_job_bundle_repo
        self.llm_artifact_gateway = llm_artifact_gateway

    async def create_batch_job_bundle(
        self,
        user: User,
        request: CreateBatchCompletionsEngineRequest,
        hardware: CreateDockerImageBatchJobResourceRequests,
    ) -> DockerImageBatchJobBundle:
        assert hardware.gpu_type is not None

        bundle_name = (
            f"{request.model_cfg.model}_{datetime.datetime.utcnow().strftime('%y%m%d-%H%M%S')}"
        )

        image_tag = await _get_latest_batch_tag(LLMInferenceFramework.VLLM)

        config_file_path = "/opt/config.json"

        batch_bundle = (
            await self.docker_image_batch_job_bundle_repo.create_docker_image_batch_job_bundle(
                name=bundle_name,
                created_by=user.user_id,
                owner=user.team_id,
                image_repository=hmi_config.batch_inference_vllm_repository,
                image_tag=image_tag,
                command=[
                    "dumb-init",
                    "--",
                    "/bin/bash",
                    "-c",
                    "ddtrace-run python vllm_batch.py",
                ],
                env={"CONFIG_FILE": config_file_path},
                mount_location=config_file_path,
                cpus=str(hardware.cpus),
                memory=str(hardware.memory),
                storage=str(hardware.storage),
                gpus=hardware.gpus,
                gpu_type=hardware.gpu_type,
                public=False,
            )
        )
        return batch_bundle

    async def execute(
        self, user: User, request: CreateBatchCompletionsV1Request
    ) -> CreateBatchCompletionsV1Response:
        if (
            request.data_parallelism is not None and request.data_parallelism > 1
        ):  # pragma: no cover
            raise ObjectHasInvalidValueException(
                "Data parallelism is disabled for batch completions."
            )

        request.model_cfg.checkpoint_path = get_checkpoint_path(
            request.model_cfg.model, request.model_cfg.checkpoint_path
        )
        hardware = await _infer_hardware(
            self.llm_artifact_gateway,
            request.model_cfg.model,
            request.model_cfg.checkpoint_path,
            is_batch_job=True,
            max_context_length=request.model_cfg.max_context_length,
        )
        assert hardware.gpus is not None

        engine_request = CreateBatchCompletionsEngineRequest.from_api_v1(request)
        engine_request.model_cfg.num_shards = hardware.gpus
        if engine_request.tool_config and engine_request.tool_config.name != "code_evaluator":
            raise ObjectHasInvalidValueException(
                "Only code_evaluator tool is supported for batch completions."
            )

        additional_engine_args = infer_addition_engine_args_from_model_name(
            engine_request.model_cfg.model
        )

        engine_request.max_gpu_memory_utilization = additional_engine_args.gpu_memory_utilization
        engine_request.attention_backend = additional_engine_args.attention_backend

        batch_bundle = await self.create_batch_job_bundle(user, engine_request, hardware)

        validate_resource_requests(
            bundle=batch_bundle,
            cpus=hardware.cpus,
            memory=hardware.memory,
            storage=hardware.storage,
            gpus=hardware.gpus,
            gpu_type=hardware.gpu_type,
        )

        if (
            engine_request.max_runtime_sec is None or engine_request.max_runtime_sec < 1
        ):  # pragma: no cover
            raise ObjectHasInvalidValueException("max_runtime_sec must be a positive integer.")

        job_id = await self.docker_image_batch_job_gateway.create_docker_image_batch_job(
            created_by=user.user_id,
            owner=user.team_id,
            job_config=engine_request.model_dump(by_alias=True),
            env=batch_bundle.env,
            command=batch_bundle.command,
            repo=batch_bundle.image_repository,
            tag=batch_bundle.image_tag,
            resource_requests=hardware,
            labels=engine_request.labels,
            mount_location=batch_bundle.mount_location,
            override_job_max_runtime_s=engine_request.max_runtime_sec,
            num_workers=engine_request.data_parallelism,
        )
        return CreateBatchCompletionsV1Response(job_id=job_id)


class CreateBatchCompletionsV2UseCase:
    def __init__(
        self,
        llm_batch_completions_service: LLMBatchCompletionsService,
        llm_artifact_gateway: LLMArtifactGateway,
    ):
        self.llm_batch_completions_service = llm_batch_completions_service
        self.llm_artifact_gateway = llm_artifact_gateway

    async def execute(
        self, request: CreateBatchCompletionsV2Request, user: User
    ) -> CreateBatchCompletionsV2Response:
        request.model_cfg.checkpoint_path = get_checkpoint_path(
            request.model_cfg.model, request.model_cfg.checkpoint_path
        )

        if (
            request.cpus is not None
            and request.gpus is not None
            and request.memory is not None
            and request.storage is not None
            and request.gpu_type is not None
        ):
            hardware = CreateDockerImageBatchJobResourceRequests(
                cpus=request.cpus,
                gpus=request.gpus,
                memory=request.memory,
                storage=request.storage,
                gpu_type=request.gpu_type,
                nodes_per_worker=request.nodes_per_worker or 1,
            )
        else:
            if (
                request.cpus is not None
                or request.gpus is not None
                or request.memory is not None
                or request.storage is not None
                or request.gpu_type is not None
            ):
                logger.warning(
                    "All hardware spec fields (cpus, gpus, memory, storage, gpu_type) must be provided if any hardware spec field is provided. Will attempt to infer hardware spec from checkpoint."
                )

            hardware = await _infer_hardware(
                self.llm_artifact_gateway,
                request.model_cfg.model,
                request.model_cfg.checkpoint_path,
                is_batch_job=True,
                max_context_length=request.model_cfg.max_context_length,
            )

        engine_request = CreateBatchCompletionsEngineRequest.from_api_v2(request)
        engine_request.model_cfg.num_shards = hardware.gpus

        validate_resource_requests(
            bundle=None,
            cpus=hardware.cpus,
            memory=hardware.memory,
            storage=hardware.storage,
            gpus=hardware.gpus,
            gpu_type=hardware.gpu_type,
        )

        if engine_request.max_runtime_sec is None or engine_request.max_runtime_sec < 1:
            raise ObjectHasInvalidValueException("max_runtime_sec must be a positive integer.")

        # Right now we only support VLLM for batch inference. Refactor this if we support more inference frameworks.
        image_repo = hmi_config.batch_inference_vllm_repository
        image_tag = await _get_latest_batch_v2_tag(LLMInferenceFramework.VLLM)

        additional_engine_args = infer_addition_engine_args_from_model_name(
            engine_request.model_cfg.model
        )

        # Overwrite model config fields with those determined by additional engine args
        for field in VLLMModelConfig.model_fields.keys():
            config_value = getattr(additional_engine_args, field, None)
            if config_value is not None and hasattr(engine_request.model_cfg, field):
                setattr(engine_request.model_cfg, field, config_value)

        engine_request.attention_backend = additional_engine_args.attention_backend

        return await self.llm_batch_completions_service.create_batch_job(
            user=user,
            job_request=engine_request,
            image_repo=image_repo,
            image_tag=image_tag,
            resource_requests=hardware,
            labels=engine_request.labels,
            max_runtime_sec=engine_request.max_runtime_sec,
            num_workers=engine_request.data_parallelism,
        )


class GetBatchCompletionV2UseCase:
    def __init__(self, llm_batch_completions_service: LLMBatchCompletionsService):
        self.llm_batch_completions_service = llm_batch_completions_service

    async def execute(
        self,
        batch_completion_id: str,
        user: User,
    ) -> GetBatchCompletionV2Response:
        job = await self.llm_batch_completions_service.get_batch_job(
            batch_completion_id,
            user=user,
        )

        if not job:
            raise ObjectNotFoundException(f"Batch completion {batch_completion_id} not found.")

        return GetBatchCompletionV2Response(job=job)


class UpdateBatchCompletionV2UseCase:
    def __init__(self, llm_batch_completions_service: LLMBatchCompletionsService):
        self.llm_batch_completions_service = llm_batch_completions_service

    async def execute(
        self,
        batch_completion_id: str,
        request: UpdateBatchCompletionsV2Request,
        user: User,
    ) -> UpdateBatchCompletionsV2Response:
        result = await self.llm_batch_completions_service.update_batch_job(
            batch_completion_id,
            user=user,
            request=request,
        )
        if not result:
            raise ObjectNotFoundException(f"Batch completion {batch_completion_id} not found.")

        return UpdateBatchCompletionsV2Response(
            **result.model_dump(by_alias=True, exclude_none=True),
            success=True,
        )


class CancelBatchCompletionV2UseCase:
    def __init__(self, llm_batch_completions_service: LLMBatchCompletionsService):
        self.llm_batch_completions_service = llm_batch_completions_service

    async def execute(
        self,
        batch_completion_id: str,
        user: User,
    ) -> CancelBatchCompletionsV2Response:
        return CancelBatchCompletionsV2Response(
            success=await self.llm_batch_completions_service.cancel_batch_job(
                batch_completion_id,
                user=user,
            )
        )
