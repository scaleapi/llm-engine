"""
DTOs for LLM APIs.
"""

import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import pydantic

if int(pydantic.__version__.split(".")[0]) > 1:
    from pydantic.v1 import BaseModel, Field, HttpUrl
else:
    from pydantic import BaseModel, Field, HttpUrl  # type: ignore

CpuSpecificationType = Union[str, int, float]
StorageSpecificationType = Union[str, int, float]  # TODO(phil): we can make this more specific.


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
    """

    cpus: CpuSpecificationType  # TODO(phil): try to use decimal.Decimal
    gpus: int = Field(..., ge=0)
    memory: StorageSpecificationType
    gpu_type: Optional[GpuType]
    storage: Optional[StorageSpecificationType]
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
    cpus: CpuSpecificationType
    gpus: int
    memory: StorageSpecificationType
    gpu_type: Optional[GpuType]
    storage: Optional[StorageSpecificationType]
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


class DeleteLLMEndpointResponse(BaseModel):
    """
    Response object for deleting a Model.
    """

    deleted: bool = Field(..., description="Whether deletion was successful.")
    """
    Whether the deletion succeeded.
    """


class CompletionSyncV1Request(BaseModel):
    """
    Request object for a synchronous prompt completion task.
    """

    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(..., gt=0)
    temperature: float = Field(..., ge=0.0)
    stop_sequences: Optional[List[str]] = Field(default=None)
    return_token_log_probs: Optional[bool] = Field(default=False)
    presence_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=-1)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)


class TokenOutput(BaseModel):
    """
    Detailed token information.
    """

    token: str
    """
    The token text.
    """

    log_prob: float
    """
    The log probability of the token.
    """


class CompletionOutput(BaseModel):
    """
    Represents the output of a completion request to a model.
    """

    text: str
    """The text of the completion."""

    # We're not guaranteed to have `num_prompt_tokens` in the response in all cases, so to be safe, set a default.
    # If we send request to api.spellbook.scale.com, we don't get this back.
    num_prompt_tokens: Optional[int] = None
    """Number of tokens in the prompt."""

    num_completion_tokens: int
    """Number of tokens in the completion."""

    tokens: Optional[List[TokenOutput]] = None
    """Detailed token information."""


class CompletionSyncResponse(BaseModel):
    """
    Response object for a synchronous prompt completion.
    """

    request_id: str
    """The unique ID of the corresponding Completion request. This `request_id` is generated on the server, and all logs 
    associated with the request are grouped by the `request_id`, which allows for easier troubleshooting of errors as
    follows:

    * When running the *Scale-hosted* LLM Engine, please provide the `request_id` in any bug reports.
    * When running the *self-hosted* LLM Engine, the `request_id` serves as a trace ID in your observability 
    provider."""

    output: CompletionOutput
    """Completion output."""


class CompletionStreamV1Request(BaseModel):
    """
    Request object for a streaming prompt completion.
    """

    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(..., gt=0)
    temperature: float = Field(..., ge=0.0)
    stop_sequences: Optional[List[str]] = Field(default=None)
    return_token_log_probs: Optional[bool] = Field(default=False)
    presence_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=-1)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)


class CompletionStreamOutput(BaseModel):
    text: str
    """The text of the completion."""

    finished: bool
    """Whether the completion is finished."""

    # We're not guaranteed to have `num_prompt_tokens` in the response in all cases, so to be safe, set a default.
    num_prompt_tokens: Optional[int] = None
    """Number of tokens in the prompt."""

    num_completion_tokens: Optional[int] = None
    """Number of tokens in the completion."""

    token: Optional[TokenOutput] = None
    """Detailed token information."""


class StreamErrorContent(BaseModel):
    error: str
    """Error message."""
    timestamp: str
    """Timestamp of the error."""


class StreamError(BaseModel):
    """
    Error object for a stream prompt completion task.
    """

    status_code: int
    """The HTTP status code of the error."""
    content: StreamErrorContent
    """The error content."""


class CompletionStreamResponse(BaseModel):
    """
    Response object for a stream prompt completion task.
    """

    request_id: str
    """The unique ID of the corresponding Completion request. This `request_id` is generated on the server, and all logs 
    associated with the request are grouped by the `request_id`, which allows for easier troubleshooting of errors as
    follows:

    * When running the *Scale-hosted* LLM Engine, please provide the `request_id` in any bug reports.
    * When running the *self-hosted* LLM Engine, the `request_id` serves as a trace ID in your observability 
    provider."""

    output: Optional[CompletionStreamOutput] = None
    """Completion output."""

    error: Optional[StreamError] = None
    """Error of the response (if any)."""


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
        ..., description="Dictionary of (file_name, url) pairs to download the model from."
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


class CreateBatchCompletionsRequestContent(BaseModel):
    prompts: List[str]
    max_new_tokens: int
    temperature: float = Field(ge=0.0, le=1.0)
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
    presence_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    """
    Only supported in vllm, lightllm
    Penalize new tokens based on whether they appear in the text so far. 0.0 means no penalty
    """
    frequency_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    """
    Only supported in vllm, lightllm
    Penalize new tokens based on their existing frequency in the text so far. 0.0 means no penalty
    """
    top_k: Optional[int] = Field(default=None, ge=-1)
    """
    Controls the number of top tokens to consider. -1 means consider all tokens.
    """
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    """
    Controls the cumulative probability of the top tokens to consider. 1.0 means consider all tokens.
    """


class CreateBatchCompletionsModelConfig(BaseModel):
    model: str
    checkpoint_path: Optional[str] = None
    """
    Path to the checkpoint to load the model from.
    """
    labels: Dict[str, str]
    """
    Labels to attach to the batch inference job.
    """
    num_shards: Optional[int] = 1
    """
    Suggested number of shards to distribute the model. When not specified, will infer the number of shards based on model config.
    System may decide to use a different number than the given value.
    """
    quantize: Optional[Quantization] = None
    """
    Whether to quantize the model.
    """
    seed: Optional[int] = None
    """
    Random seed for the model.
    """


class ToolConfig(BaseModel):
    """
    Configuration for tool use.
    NOTE: this config is highly experimental and signature will change significantly in future iterations.
    """

    name: str
    """
    Name of the tool to use for the batch inference.
    """
    max_iterations: Optional[int] = 10
    """
    Maximum number of iterations to run the tool.
    """
    execution_timeout_seconds: Optional[int] = 60
    """
    Maximum runtime of the tool in seconds.
    """
    should_retry_on_error: Optional[bool] = True
    """
    Whether to retry the tool on error.
    """


class CreateBatchCompletionsRequest(BaseModel):
    """
    Request object for batch completions.
    """

    input_data_path: Optional[str]
    output_data_path: str
    """
    Path to the output file. The output file will be a JSON file of type List[CompletionOutput].
    """
    content: Optional[CreateBatchCompletionsRequestContent] = None
    """
    Either `input_data_path` or `content` needs to be provided.
    When input_data_path is provided, the input file should be a JSON file of type BatchCompletionsRequestContent.
    """
    model_config: CreateBatchCompletionsModelConfig
    """
    Model configuration for the batch inference. Hardware configurations are inferred.
    """
    data_parallelism: Optional[int] = Field(default=1, ge=1, le=64)
    """
    Number of replicas to run the batch inference. More replicas are slower to schedule but faster to inference.
    """
    max_runtime_sec: Optional[int] = Field(default=24 * 3600, ge=1, le=2 * 24 * 3600)
    """
    Maximum runtime of the batch inference in seconds. Default to one day.
    """
    tool_config: Optional[ToolConfig] = None
    """
    Configuration for tool use.
    NOTE: this config is highly experimental and signature will change significantly in future iterations.
    """


class CreateBatchCompletionsResponse(BaseModel):
    job_id: str
    """
    The ID of the batch completions job.
    """
