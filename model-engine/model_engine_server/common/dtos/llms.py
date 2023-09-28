"""
DTOs for LLM APIs.
"""

from typing import Any, Dict, List, Optional

from model_engine_server.common.dtos.model_endpoints import (
    CpuSpecificationType,
    GetModelEndpointV1Response,
    GpuType,
    ModelEndpointType,
    StorageSpecificationType,
)
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
from pydantic import BaseModel, Field, HttpUrl


class CreateLLMModelEndpointV1Request(BaseModel):
    name: str

    # LLM specific fields
    model_name: str
    source: LLMSource = LLMSource.HUGGING_FACE
    inference_framework: LLMInferenceFramework = LLMInferenceFramework.DEEPSPEED
    inference_framework_image_tag: str
    num_shards: int = 1
    """
    Number of shards to distribute the model onto GPUs. Only affects behavior for text-generation-inference models
    """

    quantize: Optional[Quantization] = None
    """
    Whether to quantize the model. Only affect behavior for text-generation-inference models
    """

    checkpoint_path: Optional[str] = None
    """
    Path to the checkpoint to load the model from. Only affects behavior for text-generation-inference models
    """

    # General endpoint fields
    metadata: Dict[str, Any]  # TODO: JSON type
    post_inference_hooks: Optional[List[str]]
    endpoint_type: ModelEndpointType = ModelEndpointType.SYNC
    cpus: CpuSpecificationType
    gpus: int
    memory: StorageSpecificationType
    gpu_type: GpuType
    storage: Optional[StorageSpecificationType]
    optimize_costs: Optional[bool]
    min_workers: int
    max_workers: int
    per_worker: int
    labels: Dict[str, str]
    prewarm: Optional[bool]
    high_priority: Optional[bool]
    billing_tags: Optional[Dict[str, Any]]
    default_callback_url: Optional[HttpUrl]
    default_callback_auth: Optional[CallbackAuth]
    public_inference: Optional[bool] = True  # LLM endpoints are public by default.


class CreateLLMModelEndpointV1Response(BaseModel):
    endpoint_creation_task_id: str


class GetLLMModelEndpointV1Response(BaseModel):
    id: str
    """
    The autogenerated ID of the Launch endpoint.
    """

    name: str
    model_name: Optional[str] = None
    source: LLMSource
    status: ModelEndpointStatus
    inference_framework: LLMInferenceFramework
    inference_framework_image_tag: Optional[str] = None
    num_shards: Optional[int] = None
    quantize: Optional[Quantization] = None
    spec: Optional[GetModelEndpointV1Response] = None


class ListLLMModelEndpointsV1Response(BaseModel):
    model_endpoints: List[GetLLMModelEndpointV1Response]


# Delete and update use the default Launch endpoint APIs.


class CompletionSyncV1Request(BaseModel):
    """
    Request object for a synchronous prompt completion task.
    """

    prompt: str
    max_new_tokens: int
    temperature: float = Field(ge=0, le=1)
    """
    Temperature of the sampling. Setting to 0 equals to greedy sampling.
    """
    stop_sequences: Optional[List[str]] = None
    """
    List of sequences to stop the completion at.
    """
    return_token_log_probs: Optional[bool] = False
    """
    Whether to return the log probabilities of the tokens.
    """
    repetition_penalty: Optional[float] = None
    """
    The parameter for repetition penalty. 1.0 means no penalty. 
    """
    top_k: Optional[int] = None
    """
    Integer that controls the number of top tokens to consider.
    """
    top_p: Optional[float] = None
    """
    Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1].
    """


class TokenOutput(BaseModel):
    token: str
    log_prob: float


class CompletionOutput(BaseModel):
    text: str
    num_completion_tokens: int
    tokens: Optional[List[TokenOutput]] = None


class CompletionSyncV1Response(BaseModel):
    """
    Response object for a synchronous prompt completion task.
    """

    request_id: str
    output: Optional[CompletionOutput] = None


class CompletionStreamV1Request(BaseModel):
    """
    Request object for a stream prompt completion task.
    """

    prompt: str
    max_new_tokens: int
    temperature: float = Field(ge=0, le=1)
    """
    Temperature of the sampling. Setting to 0 equals to greedy sampling.
    """
    stop_sequences: Optional[List[str]] = None
    """
    List of sequences to stop the completion at.
    """
    return_token_log_probs: Optional[bool] = False
    """
    Whether to return the log probabilities of the tokens. Only affects behavior for text-generation-inference models
    """
    repetition_penalty: Optional[float] = None
    """
    The parameter for repetition penalty. 1.0 means no penalty. 
    """
    top_k: Optional[int] = None
    """
    Integer that controls the number of top tokens to consider.
    """
    top_p: Optional[float] = None
    """
    Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1].
    """


class CompletionStreamOutput(BaseModel):
    text: str
    finished: bool
    num_completion_tokens: Optional[int] = None
    token: Optional[TokenOutput] = None


class CompletionStreamV1Response(BaseModel):
    """
    Response object for a stream prompt completion task.
    """

    request_id: str
    output: Optional[CompletionStreamOutput] = None


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
        ..., description="Dictionary of (file_name, url) pairs to download the model from."
    )


class DeleteLLMEndpointResponse(BaseModel):
    deleted: bool
