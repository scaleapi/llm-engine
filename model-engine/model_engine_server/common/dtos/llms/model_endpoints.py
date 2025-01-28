"""
DTOs for LLM APIs.

"""

from typing import Any, Dict, List, Literal, Optional, TypeAlias, Union

from model_engine_server.common.dtos.core import HttpUrlStr
from model_engine_server.common.dtos.llms.sglang import SGLangEndpointAdditionalArgs
from model_engine_server.common.dtos.llms.vllm import VLLMEndpointAdditionalArgs
from model_engine_server.common.dtos.model_endpoints import (
    CpuSpecificationType,
    GetModelEndpointV1Response,
    GpuType,
    ModelEndpointType,
    StorageSpecificationType,
)
from model_engine_server.common.pydantic_types import BaseModel, Field
from model_engine_server.domain.entities import (
    BatchJobStatus,
    CallbackAuth,
    FineTuneHparamValueType,
    LLMFineTuneEvent,
    LLMInferenceFramework,
    LLMSource,
    ModelEndpointStatus,
    Quantization,
)
from pydantic import Discriminator, Tag
from typing_extensions import Annotated


class LLMModelEndpointCommonArgs(BaseModel):
    quantize: Optional[Quantization] = None
    """
    Whether to quantize the model.
    """

    checkpoint_path: Optional[str] = None
    """
    Path to the checkpoint to load the model from.
    """

    # General endpoint fields
    post_inference_hooks: Optional[List[str]] = None
    cpus: Optional[CpuSpecificationType] = None
    gpus: Optional[int] = None
    memory: Optional[StorageSpecificationType] = None
    gpu_type: Optional[GpuType] = None
    storage: Optional[StorageSpecificationType] = None
    nodes_per_worker: Optional[int] = None
    optimize_costs: Optional[bool] = None
    prewarm: Optional[bool] = None
    high_priority: Optional[bool] = None
    billing_tags: Optional[Dict[str, Any]] = None
    default_callback_url: Optional[HttpUrlStr] = None
    default_callback_auth: Optional[CallbackAuth] = None
    public_inference: Optional[bool] = True  # LLM endpoints are public by default.
    chat_template_override: Optional[str] = Field(
        default=None,
        description="A Jinja template to use for this endpoint. If not provided, will use the chat template from the checkpoint",
    )


class CreateLLMModelEndpointArgs(LLMModelEndpointCommonArgs):
    name: str
    model_name: str
    source: LLMSource = LLMSource.HUGGING_FACE
    inference_framework: LLMInferenceFramework = LLMInferenceFramework.VLLM
    inference_framework_image_tag: str = "latest"
    num_shards: int = 1
    """
    Number of shards to distribute the model onto GPUs.
    """
    metadata: Dict[str, Any]  # TODO: JSON type
    endpoint_type: ModelEndpointType = ModelEndpointType.SYNC
    min_workers: int
    max_workers: int
    per_worker: int
    labels: Dict[str, str]


class CreateVLLMModelEndpointRequest(
    VLLMEndpointAdditionalArgs, CreateLLMModelEndpointArgs, BaseModel
):
    inference_framework: Literal[LLMInferenceFramework.VLLM] = LLMInferenceFramework.VLLM
    pass


class CreateSGLangModelEndpointRequest(
    SGLangEndpointAdditionalArgs, CreateLLMModelEndpointArgs, BaseModel
):
    inference_framework: Literal[LLMInferenceFramework.SGLANG] = LLMInferenceFramework.SGLANG
    pass


class CreateDeepSpeedModelEndpointRequest(CreateLLMModelEndpointArgs, BaseModel):
    inference_framework: Literal[LLMInferenceFramework.DEEPSPEED] = LLMInferenceFramework.DEEPSPEED
    pass


class CreateTextGenerationInferenceModelEndpointRequest(CreateLLMModelEndpointArgs, BaseModel):
    inference_framework: Literal[LLMInferenceFramework.TEXT_GENERATION_INFERENCE] = (
        LLMInferenceFramework.TEXT_GENERATION_INFERENCE
    )
    pass


class CreateLightLLMModelEndpointRequest(CreateLLMModelEndpointArgs, BaseModel):
    inference_framework: Literal[LLMInferenceFramework.LIGHTLLM] = LLMInferenceFramework.LIGHTLLM
    pass


class CreateTensorRTLLMModelEndpointRequest(CreateLLMModelEndpointArgs, BaseModel):
    inference_framework: Literal[LLMInferenceFramework.TENSORRT_LLM] = (
        LLMInferenceFramework.TENSORRT_LLM
    )
    pass


def get_inference_framework(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("inference_framework", LLMInferenceFramework.VLLM)
    return getattr(v, "inference_framework", LLMInferenceFramework.VLLM)


CreateLLMModelEndpointV1Request: TypeAlias = Annotated[
    Union[
        Annotated[CreateVLLMModelEndpointRequest, Tag(LLMInferenceFramework.VLLM)],
        Annotated[CreateSGLangModelEndpointRequest, Tag(LLMInferenceFramework.SGLANG)],
        Annotated[CreateDeepSpeedModelEndpointRequest, Tag(LLMInferenceFramework.DEEPSPEED)],
        Annotated[
            CreateTextGenerationInferenceModelEndpointRequest,
            Tag(LLMInferenceFramework.TEXT_GENERATION_INFERENCE),
        ],
        Annotated[CreateLightLLMModelEndpointRequest, Tag(LLMInferenceFramework.LIGHTLLM)],
        Annotated[CreateTensorRTLLMModelEndpointRequest, Tag(LLMInferenceFramework.TENSORRT_LLM)],
    ],
    Discriminator(get_inference_framework),
]


class CreateLLMModelEndpointV1Response(BaseModel):
    endpoint_creation_task_id: str


class GetLLMModelEndpointV1Response(BaseModel):
    id: str
    """
    The autogenerated ID of the Launch endpoint.
    """

    name: str
    model_name: str
    source: LLMSource
    status: ModelEndpointStatus
    inference_framework: LLMInferenceFramework
    inference_framework_image_tag: Optional[str] = None
    num_shards: Optional[int] = None
    quantize: Optional[Quantization] = None
    checkpoint_path: Optional[str] = None
    chat_template_override: Optional[str] = Field(
        default=None,
        description="A Jinja template to use for this endpoint. If not provided, will use the chat template from the checkpoint",
    )
    spec: Optional[GetModelEndpointV1Response] = None


class ListLLMModelEndpointsV1Response(BaseModel):
    model_endpoints: List[GetLLMModelEndpointV1Response]


class UpdateLLMModelEndpointArgs(LLMModelEndpointCommonArgs):
    model_name: Optional[str] = None
    source: Optional[LLMSource] = None
    inference_framework: Optional[LLMInferenceFramework] = None
    inference_framework_image_tag: Optional[str] = None
    num_shards: Optional[int] = None
    """
    Number of shards to distribute the model onto GPUs.
    """
    metadata: Optional[Dict[str, Any]] = None
    force_bundle_recreation: Optional[bool] = False
    """
    Whether to force recreate the underlying bundle.

    If True, the underlying bundle will be recreated. This is useful if there are underlying implementation changes with how bundles are created
    that we would like to pick up for existing endpoints
    """
    min_workers: Optional[int] = None
    max_workers: Optional[int] = None
    per_worker: Optional[int] = None
    labels: Optional[Dict[str, str]] = None


class UpdateVLLMModelEndpointRequest(
    VLLMEndpointAdditionalArgs, UpdateLLMModelEndpointArgs, BaseModel
):
    inference_framework: Literal[LLMInferenceFramework.VLLM] = LLMInferenceFramework.VLLM


class UpdateSGLangModelEndpointRequest(
    SGLangEndpointAdditionalArgs, UpdateLLMModelEndpointArgs, BaseModel
):
    inference_framework: Literal[LLMInferenceFramework.SGLANG] = LLMInferenceFramework.SGLANG


class UpdateDeepSpeedModelEndpointRequest(UpdateLLMModelEndpointArgs, BaseModel):
    inference_framework: Literal[LLMInferenceFramework.DEEPSPEED] = LLMInferenceFramework.DEEPSPEED


class UpdateTextGenerationInferenceModelEndpointRequest(UpdateLLMModelEndpointArgs, BaseModel):
    inference_framework: Literal[LLMInferenceFramework.TEXT_GENERATION_INFERENCE] = (
        LLMInferenceFramework.TEXT_GENERATION_INFERENCE
    )


class UpdateLightLLMModelEndpointRequest(UpdateLLMModelEndpointArgs, BaseModel):
    inference_framework: Literal[LLMInferenceFramework.LIGHTLLM] = LLMInferenceFramework.LIGHTLLM


class UpdateTensorRTLLMModelEndpointRequest(UpdateLLMModelEndpointArgs, BaseModel):
    inference_framework: Literal[LLMInferenceFramework.TENSORRT_LLM] = (
        LLMInferenceFramework.TENSORRT_LLM
    )


UpdateLLMModelEndpointV1Request: TypeAlias = Annotated[
    Union[
        Annotated[UpdateVLLMModelEndpointRequest, Tag(LLMInferenceFramework.VLLM)],
        Annotated[UpdateSGLangModelEndpointRequest, Tag(LLMInferenceFramework.SGLANG)],
        Annotated[UpdateDeepSpeedModelEndpointRequest, Tag(LLMInferenceFramework.DEEPSPEED)],
        Annotated[
            UpdateTextGenerationInferenceModelEndpointRequest,
            Tag(LLMInferenceFramework.TEXT_GENERATION_INFERENCE),
        ],
    ],
    Discriminator(get_inference_framework),
]


class UpdateLLMModelEndpointV1Response(BaseModel):
    endpoint_creation_task_id: str


class CreateFineTuneRequest(BaseModel):
    model: str
    training_file: str
    validation_file: Optional[str] = None
    # fine_tuning_method: str  # TODO enum + uncomment when we support multiple methods
    hyperparameters: Dict[str, FineTuneHparamValueType]  # validated somewhere else
    suffix: Optional[str] = None
    wandb_config: Optional[Dict[str, Any]] = None
    """
    Config to pass to wandb for init. See https://docs.wandb.ai/ref/python/init
    Must include `api_key` field which is the wandb API key.
    """


class CreateFineTuneResponse(BaseModel):
    id: str


class GetFineTuneResponse(BaseModel):
    id: str = Field(..., description="Unique ID of the fine tune")
    fine_tuned_model: Optional[str] = Field(
        default=None,
        description="Name of the resulting fine-tuned model. This can be plugged into the "
        "Completion API ones the fine-tune is complete",
    )
    status: BatchJobStatus = Field(..., description="Status of the requested fine tune.")


class ListFineTunesResponse(BaseModel):
    jobs: List[GetFineTuneResponse]


class CancelFineTuneResponse(BaseModel):
    success: bool


class GetFineTuneEventsResponse(BaseModel):
    # LLMFineTuneEvent is entity layer technically, but it's really simple
    events: List[LLMFineTuneEvent]


class ModelDownloadRequest(BaseModel):
    model_name: str = Field(..., description="Name of the fine tuned model")
    download_format: Optional[str] = Field(
        default="hugging_face",
        description="Format that you want the downloaded urls to be compatible with. Currently only supports hugging_face",
    )


class ModelDownloadResponse(BaseModel):
    urls: Dict[str, str] = Field(
        ...,
        description="Dictionary of (file_name, url) pairs to download the model from.",
    )


# Delete uses the default Launch endpoint APIs.
class DeleteLLMEndpointResponse(BaseModel):
    deleted: bool
