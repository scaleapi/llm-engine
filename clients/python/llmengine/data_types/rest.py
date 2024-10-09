"""
DTOs for LLM APIs.
"""

import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from .pydantic_types import BaseModel, Field, HttpUrl

CpuSpecificationType = Union[str, int, float]
StorageSpecificationType = Union[str, int, float]


class LLMInferenceFramework(str, Enum):
    DEEPSPEED = "deepspeed"
    TEXT_GENERATION_INFERENCE = "text_generation_inference"
    VLLM = "vllm"
    LIGHTLLM = "lightllm"
    TENSORRT_LLM = "tensorrt_llm"


class LLMSource(str, Enum):
    HUGGING_FACE = "hugging_face"


class Quantization(str, Enum):
    BITSANDBYTES = "bitsandbytes"
    AWQ = "awq"


class GpuType(str, Enum):
    """Lists allowed GPU types for LLMEngine."""

    NVIDIA_TESLA_T4 = "nvidia-tesla-t4"
    NVIDIA_AMPERE_A10 = "nvidia-ampere-a10"
    NVIDIA_AMPERE_A100 = "nvidia-ampere-a100"
    NVIDIA_AMPERE_A100E = "nvidia-ampere-a100e"
    NVIDIA_HOPPER_H100 = "nvidia-hopper-h100"
    NVIDIA_HOPPER_H100_1G_20GB = "nvidia-hopper-h100-1g20gb"
    NVIDIA_HOPPER_H100_3G_40GB = "nvidia-hopper-h100-3g40gb"


class ModelEndpointType(str, Enum):
    STREAMING = "streaming"


class ModelEndpointStatus(str, Enum):
    # Duplicates common/types::EndpointStatus, when refactor is done, delete the old type
    # See EndpointStatus for status explanations
    READY = "READY"
    UPDATE_PENDING = "UPDATE_PENDING"
    UPDATE_IN_PROGRESS = "UPDATE_IN_PROGRESS"
    UPDATE_FAILED = "UPDATE_FAILED"
    DELETE_IN_PROGRESS = "DELETE_IN_PROGRESS"


class CallbackBasicAuth(BaseModel):
    kind: Literal["basic"]
    username: str
    password: str


class CallbackmTLSAuth(BaseModel):
    kind: Literal["mtls"]
    cert: str
    key: str


class CallbackAuth(BaseModel):
    __root__: Union[CallbackBasicAuth, CallbackmTLSAuth] = Field(..., discriminator="kind")


class ModelEndpointDeploymentState(BaseModel):
    """
    This is the entity-layer class for the deployment settings related to a Model Endpoint.
    """

    min_workers: int = Field(..., ge=0)
    max_workers: int = Field(..., ge=0)
    per_worker: int = Field(..., gt=0)
    available_workers: Optional[int] = Field(default=None, ge=0)
    unavailable_workers: Optional[int] = Field(default=None, ge=0)


class ModelEndpointResourceState(BaseModel):
    """
    This is the entity-layer class for the resource settings per worker of a Model Endpoint.
    Note: the values for cpus/gpus/memory/storage are per node, i.e. a single "worker" may consist of
    multiple underlying "nodes" (corresponding to kubernetes pods), and the values for cpus/gpus/memory/storage
    are the resources allocated for a single node. Thus, the total resource allocation
    for the entire worker is multiplied by the value of `nodes_per_worker`.
    """

    cpus: CpuSpecificationType  # TODO(phil): try to use decimal.Decimal
    gpus: int = Field(..., ge=0)
    memory: StorageSpecificationType
    gpu_type: Optional[GpuType]
    storage: Optional[StorageSpecificationType]
    nodes_per_worker: int = Field(..., ge=1)  # Multinode support. >1 = multinode.
    optimize_costs: Optional[bool]


class GetModelEndpointResponse(BaseModel):
    id: str
    name: str
    endpoint_type: ModelEndpointType
    destination: str
    deployment_name: Optional[str] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)  # TODO: JSON type
    bundle_name: str
    status: ModelEndpointStatus
    post_inference_hooks: Optional[List[str]] = Field(default=None)
    default_callback_url: Optional[HttpUrl] = Field(default=None)
    default_callback_auth: Optional[CallbackAuth] = Field(default=None)
    labels: Optional[Dict[str, str]] = Field(default=None)
    aws_role: Optional[str] = Field(default=None)
    results_s3_bucket: Optional[str] = Field(default=None)
    created_by: str
    created_at: datetime.datetime
    last_updated_at: datetime.datetime
    deployment_state: Optional[ModelEndpointDeploymentState] = Field(default=None)
    resource_state: Optional[ModelEndpointResourceState] = Field(default=None)
    num_queued_items: Optional[int] = Field(default=None)
    public_inference: Optional[bool] = Field(default=None)


class PostInferenceHooks(str, Enum):
    """
    Post-inference hooks are functions that are called after inference is complete.

    Attributes:
        CALLBACK: The callback hook is called with the inference response and the task ID.
    """

    # INSIGHT = "insight"
    CALLBACK: str = "callback"


class CreateLLMEndpointRequest(BaseModel):
    name: str

    # LLM specific fields
    model_name: str
    source: LLMSource = LLMSource.HUGGING_FACE
    inference_framework: LLMInferenceFramework = LLMInferenceFramework.VLLM
    inference_framework_image_tag: str
    num_shards: int = 1
    """
    Number of shards to distribute the model onto GPUs. Only affects behavior for text-generation-inference models
    """

    quantize: Optional[Quantization] = None
    """
    Quantization for the LLM. Only affects behavior for text-generation-inference models
    """

    checkpoint_path: Optional[str] = None
    """
    Path to the checkpoint to load the model from. Only affects behavior for text-generation-inference models
    """

    # General endpoint fields
    metadata: Dict[str, Any]  # TODO: JSON type
    post_inference_hooks: Optional[List[str]]
    endpoint_type: ModelEndpointType = ModelEndpointType.STREAMING
    cpus: Optional[CpuSpecificationType]
    gpus: Optional[int]
    memory: Optional[StorageSpecificationType]
    gpu_type: Optional[GpuType]
    storage: Optional[StorageSpecificationType]
    nodes_per_worker: Optional[int] = None
    optimize_costs: Optional[bool] = None
    min_workers: int
    max_workers: int
    per_worker: int
    labels: Dict[str, str]
    prewarm: Optional[bool] = None
    high_priority: Optional[bool]
    default_callback_url: Optional[HttpUrl] = None
    default_callback_auth: Optional[CallbackAuth] = None
    public_inference: Optional[bool] = True
    """
    Whether the endpoint can be used for inference for all users. LLM endpoints are public by default.
    """
    chat_template_override: Optional[str] = Field(
        default=None,
        description="A Jinja template to use for this endpoint. If not provided, will use the chat template from the checkpoint",
    )


class CreateLLMEndpointResponse(BaseModel):
    endpoint_creation_task_id: str


class GetLLMEndpointResponse(BaseModel):
    """
    Response object for retrieving a Model.
    """

    id: Optional[str] = Field(
        default=None,
        description="(For self-hosted users) The autogenerated ID of the model.",
    )
    """(For self-hosted users) The autogenerated ID of the model."""

    name: str = Field(
        description="The name of the model. Use this for making inference requests to the model."
    )
    """The name of the model. Use this for making inference requests to the model."""

    model_name: Optional[str] = Field(
        default=None,
        description="(For self-hosted users) For fine-tuned models, the base model. For base models, this will be the same as `name`.",
    )
    """(For self-hosted users) For fine-tuned models, the base model. For base models, this will be the same as `name`."""

    source: LLMSource = Field(description="The source of the model, e.g. Hugging Face.")
    """The source of the model, e.g. Hugging Face."""

    status: ModelEndpointStatus = Field(description="The status of the model.")
    """The status of the model (can be one of "READY", "UPDATE_PENDING", "UPDATE_IN_PROGRESS", "UPDATE_FAILED", "DELETE_IN_PROGRESS")."""

    inference_framework: LLMInferenceFramework = Field(
        description="The inference framework used by the model."
    )
    """(For self-hosted users) The inference framework used by the model."""

    inference_framework_tag: Optional[str] = Field(
        default=None,
        description="(For self-hosted users) The Docker image tag used to run the model.",
    )
    """(For self-hosted users) The Docker image tag used to run the model."""

    num_shards: Optional[int] = Field(
        default=None, description="(For self-hosted users) The number of shards."
    )
    """(For self-hosted users) The number of shards."""

    quantize: Optional[Quantization] = Field(
        default=None, description="(For self-hosted users) The quantization method."
    )
    """(For self-hosted users) The quantization method."""

    spec: Optional[GetModelEndpointResponse] = Field(
        default=None, description="(For self-hosted users) Model endpoint details."
    )
    """(For self-hosted users) Model endpoint details."""

    chat_template_override: Optional[str] = Field(
        default=None,
        description="A Jinja template to use for this endpoint. If not provided, will use the chat template from the checkpoint",
    )


class ListLLMEndpointsResponse(BaseModel):
    """
    Response object for listing Models.
    """

    model_endpoints: List[GetLLMEndpointResponse] = Field(
        ...,
        description="The list of models.",
    )
    """
    A list of Models, represented as `GetLLMEndpointResponse`s.
    """


class UpdateLLMEndpointRequest(BaseModel):
    # LLM specific fields
    model_name: Optional[str]
    source: Optional[LLMSource]
    inference_framework_image_tag: Optional[str]
    num_shards: Optional[int]
    """
    Number of shards to distribute the model onto GPUs.
    """

    quantize: Optional[Quantization]
    """
    Whether to quantize the model.
    """

    checkpoint_path: Optional[str]
    """
    Path to the checkpoint to load the model from.
    """

    # General endpoint fields
    metadata: Optional[Dict[str, Any]]
    post_inference_hooks: Optional[List[str]]
    cpus: Optional[CpuSpecificationType]
    gpus: Optional[int]
    memory: Optional[StorageSpecificationType]
    gpu_type: Optional[GpuType]
    storage: Optional[StorageSpecificationType]
    optimize_costs: Optional[bool]
    min_workers: Optional[int]
    max_workers: Optional[int]
    per_worker: Optional[int]
    labels: Optional[Dict[str, str]]
    prewarm: Optional[bool]
    high_priority: Optional[bool]
    billing_tags: Optional[Dict[str, Any]]
    default_callback_url: Optional[HttpUrl]
    default_callback_auth: Optional[CallbackAuth]
    public_inference: Optional[bool]
    chat_template_override: Optional[str] = Field(
        default=None,
        description="A Jinja template to use for this endpoint. If not provided, will use the chat template from the checkpoint",
    )

    force_bundle_recreation: Optional[bool] = False
    """
    Whether to force recreate the underlying bundle.

    If True, the underlying bundle will be recreated. This is useful if there are underlying implementation changes with how bundles are created
    that we would like to pick up for existing endpoints
    """


class UpdateLLMEndpointResponse(BaseModel):
    endpoint_creation_task_id: str


class DeleteLLMEndpointResponse(BaseModel):
    """
    Response object for deleting a Model.
    """

    deleted: bool = Field(..., description="Whether deletion was successful.")
    """
    Whether the deletion succeeded.
    """


class CreateFineTuneRequest(BaseModel):
    """
    Request object for creating a FineTune.
    """

    model: str = Field(..., description="Identifier of base model to train from.")
    """Identifier of base model to train from."""

    training_file: str = Field(
        ...,
        description="Path to file of training dataset. Dataset must be a csv with columns 'prompt' and 'response'.",
    )
    """Path to file of training dataset. Dataset must be a csv with columns 'prompt' and 'response'."""

    validation_file: Optional[str] = Field(
        default=None,
        description="Path to file of validation dataset. Has the same format as training_file. If not provided, we will generate a split from the training dataset.",
    )
    """Path to file of validation dataset. Has the same format as training_file. If not provided, we will generate a split from the training dataset."""

    hyperparameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Hyperparameters to pass in to training job."
    )
    """Hyperparameters to pass in to training job."""

    wandb_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Configuration for Weights and Biases."
    )
    """
    A dict of configuration parameters for Weights & Biases. See [Weights & Biases](https://docs.wandb.ai/ref/python/init) for more information.
    Set `hyperparameter["report_to"]` to `wandb` to enable automatic finetune metrics logging.
    Must include `api_key` field which is the wandb API key.
    Also supports setting `base_url` to use a custom Weights & Biases server.
    """

    suffix: Optional[str] = Field(
        default=None,
        description="Optional user-provided identifier suffix for the fine-tuned model. Can be up to 28 characters long.",
    )
    """Optional user-provided identifier suffix for the fine-tuned model. Can be up to 28 characters long."""


class CreateFineTuneResponse(BaseModel):
    """
    Response object for creating a FineTune.
    """

    id: str = Field(..., description="ID of the created fine-tuning job.")
    """
    The ID of the FineTune.
    """


class BatchJobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELLED = "CANCELLED"
    UNDEFINED = "UNDEFINED"
    TIMEOUT = "TIMEOUT"


class GetFineTuneResponse(BaseModel):
    """
    Response object for retrieving a FineTune.
    """

    id: str = Field(..., description="ID of the requested job.")
    """
    The ID of the FineTune.
    """

    fine_tuned_model: Optional[str] = Field(
        default=None,
        description="Name of the resulting fine-tuned model. This can be plugged into the "
        "Completion API once the fine-tune is complete",
    )
    """
    The name of the resulting fine-tuned model. This can be plugged into the Completion API
    once the fine-tune is complete.
    """

    status: BatchJobStatus = Field(..., description="Status of the requested job.")
    """
    The status of the FineTune job.
    """


class ListFineTunesResponse(BaseModel):
    """
    Response object for listing FineTunes.
    """

    jobs: List[GetFineTuneResponse] = Field(
        ..., description="List of fine-tuning jobs and their statuses."
    )
    """
    A list of FineTunes, represented as `GetFineTuneResponse`s.
    """


class CancelFineTuneResponse(BaseModel):
    """
    Response object for cancelling a FineTune.
    """

    success: bool = Field(..., description="Whether cancellation was successful.")
    """
    Whether the cancellation succeeded.
    """


class LLMFineTuneEvent(BaseModel):
    """
    Response object one FineTune event.
    """

    timestamp: Optional[float] = Field(
        description="Timestamp of the event.",
        default=None,
    )
    message: str = Field(description="Message of the event.")
    level: str = Field(description="Logging level of the event.")


class GetFineTuneEventsResponse(BaseModel):
    """
    Response object for getting events for a FineTune.
    """

    events: List[LLMFineTuneEvent] = Field(..., description="List of fine-tuning events.")


class ModelDownloadRequest(BaseModel):
    """
    Request object for downloading a model.
    """

    model_name: str = Field(..., description="Name of the model to download.")
    download_format: Optional[str] = Field(
        default="hugging_face",
        description="Desired return format for downloaded model weights (default=hugging_face).",
    )


class ModelDownloadResponse(BaseModel):
    """
    Response object for downloading a model.
    """

    urls: Dict[str, str] = Field(
        ...,
        description="Dictionary of (file_name, url) pairs to download the model from.",
    )


class UploadFileResponse(BaseModel):
    """Response object for uploading a file."""

    id: str = Field(..., description="ID of the uploaded file.")
    """ID of the uploaded file."""


class GetFileResponse(BaseModel):
    """Response object for retrieving a file."""

    id: str = Field(..., description="ID of the requested file.")
    """ID of the requested file."""

    filename: str = Field(..., description="File name.")
    """File name."""

    size: int = Field(..., description="Length of the file, in characters.")
    """Length of the file, in characters."""


class ListFilesResponse(BaseModel):
    """Response object for listing files."""

    files: List[GetFileResponse] = Field(..., description="List of file IDs, names, and sizes.")
    """List of file IDs, names, and sizes."""


class DeleteFileResponse(BaseModel):
    """Response object for deleting a file."""

    deleted: bool = Field(..., description="Whether deletion was successful.")
    """Whether deletion was successful."""


class GetFileContentResponse(BaseModel):
    """Response object for retrieving a file's content."""

    id: str = Field(..., description="ID of the requested file.")
    """ID of the requested file."""

    content: str = Field(..., description="File content.")
    """File content."""
