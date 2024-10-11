from enum import Enum
from typing import Dict, List, Optional, Union

from typing_extensions import TypeAlias

from .chat_completion import ChatCompletionV2Request, ChatCompletionV2Response
from .completion import CompletionOutput, CompletionV2Request, CompletionV2Response
from .pydantic_types import BaseModel, Field
from .rest import CpuSpecificationType, GpuType, StorageSpecificationType


# Common DTOs for batch completions
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


class BatchCompletionsModelConfig(BaseModel):
    model: str = Field(
        description="ID of the model to use.",
        examples=["mixtral-8x7b-instruct"],
    )

    checkpoint_path: Optional[str] = Field(
        default=None, description="Path to the checkpoint to load the model from."
    )

    num_shards: Optional[int] = Field(
        default=1,
        ge=1,
        description="""
Suggested number of shards to distribute the model. When not specified, will infer the number of shards based on model config.
System may decide to use a different number than the given value.
""",
    )

    max_context_length: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum context length to use for the model. Defaults to the max allowed by the model",
    )

    seed: Optional[int] = Field(default=None, description="Random seed for the model.")

    response_role: Optional[str] = Field(
        default=None,
        description="Role of the response in the conversation. Only supported in chat completions.",
    )


class BatchCompletionsRequestBase(BaseModel):
    input_data_path: Optional[str] = Field(
        default=None,
        description="Path to the input file. The input file should be a JSON file of type List[CreateBatchCompletionsRequestContent].",
    )
    output_data_path: str = Field(
        description="Path to the output file. The output file will be a JSON file of type List[CompletionOutput]."
    )

    labels: Dict[str, str] = Field(
        default={}, description="Labels to attach to the batch inference job."
    )

    data_parallelism: Optional[int] = Field(
        default=1,
        ge=1,
        le=64,
        description="Number of replicas to run the batch inference. More replicas are slower to schedule but faster to inference.",
    )

    max_runtime_sec: Optional[int] = Field(
        default=24 * 3600,
        ge=1,
        le=2 * 24 * 3600,
        description="Maximum runtime of the batch inference in seconds. Default to one day.",
    )

    priority: Optional[str] = Field(
        default=None,
        description="Priority of the batch inference job. Default to None.",
    )

    tool_config: Optional[ToolConfig] = Field(
        default=None,
        description="""
Configuration for tool use.
NOTE: this config is highly experimental and signature will change significantly in future iterations.""",
    )

    cpus: Optional[CpuSpecificationType] = Field(
        default=None, description="CPUs to use for the batch inference."
    )
    gpus: Optional[int] = Field(
        default=None, description="Number of GPUs to use for the batch inference."
    )
    memory: Optional[StorageSpecificationType] = Field(
        default=None, description="Amount of memory to use for the batch inference."
    )
    gpu_type: Optional[GpuType] = Field(
        default=None, description="GPU type to use for the batch inference."
    )
    storage: Optional[StorageSpecificationType] = Field(
        default=None, description="Storage to use for the batch inference."
    )
    nodes_per_worker: Optional[int] = Field(
        default=None, description="Number of nodes per worker for the batch inference."
    )


# V1 DTOs for batch completions
CompletionV1Output = CompletionOutput


class CreateBatchCompletionsV1ModelConfig(BatchCompletionsModelConfig):
    labels: Dict[str, str] = Field(
        default={}, description="Labels to attach to the batch inference job."
    )


class CreateBatchCompletionsV1RequestContent(BaseModel):
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
    skip_special_tokens: Optional[bool] = True
    """
    Whether to skip special tokens in the output.
    """


class CreateBatchCompletionsV1Request(BatchCompletionsRequestBase):
    """
    Request object for batch completions.
    """

    content: Optional[CreateBatchCompletionsV1RequestContent] = None
    """
    Either `input_data_path` or `content` needs to be provided.
    When input_data_path is provided, the input file should be a JSON file of type BatchCompletionsRequestContent.
    """
    model_config: CreateBatchCompletionsV1ModelConfig = Field(alias="model_config")
    """
    Model configuration for the batch inference. Hardware configurations are inferred.
    """


class CreateBatchCompletionsV1Response(BaseModel):
    job_id: str


class FilteredCompletionV2Request(CompletionV2Request):
    model: Optional[str] = None  # type: ignore[assignment]
    stream: Optional[bool] = False


class FilteredChatCompletionV2Request(ChatCompletionV2Request):
    model: Optional[str] = None  # type: ignore[assignment]
    stream: Optional[bool] = False


# V2 DTOs for batch completions
CompletionRequest: TypeAlias = Union[FilteredCompletionV2Request, FilteredChatCompletionV2Request]
CompletionResponse: TypeAlias = Union[CompletionV2Response, ChatCompletionV2Response]
CreateBatchCompletionsV2RequestContent: TypeAlias = Union[
    List[FilteredCompletionV2Request], List[FilteredChatCompletionV2Request]
]
CreateBatchCompletionsV2ModelConfig: TypeAlias = BatchCompletionsModelConfig

BatchCompletionContent = Union[
    CreateBatchCompletionsV1RequestContent, CreateBatchCompletionsV2RequestContent
]


class CreateBatchCompletionsV2Request(BatchCompletionsRequestBase):
    """
    Request object for batch completions.
    """

    content: Optional[BatchCompletionContent] = Field(
        default=None,
        description="""
Either `input_data_path` or `content` needs to be provided.
When input_data_path is provided, the input file should be a JSON file of type List[CreateBatchCompletionsRequestContent].
""",
    )

    model_config: BatchCompletionsModelConfig = Field(
        description="""Model configuration for the batch inference. Hardware configurations are inferred.""",
    )


class BatchCompletionsJobStatus(str, Enum):
    Queued = "queued"
    Running = "running"
    Completed = "completed"
    Failed = "failed"
    Cancelled = "cancelled"
    Unknown = "unknown"


class BatchCompletionsJob(BaseModel):
    job_id: str
    input_data_path: Optional[str] = Field(
        default=None,
        description="Path to the input file. The input file should be a JSON file of type List[CreateBatchCompletionsRequestContent].",
    )
    output_data_path: str = Field(
        description="Path to the output file. The output file will be a JSON file of type List[CompletionOutput]."
    )

    model_config: BatchCompletionsModelConfig = Field(
        description="""Model configuration for the batch inference. Hardware configurations are inferred.""",
    )

    priority: Optional[str] = Field(
        default=None,
        description="Priority of the batch inference job. Default to None.",
    )
    status: BatchCompletionsJobStatus
    created_at: str
    expires_at: str
    completed_at: Optional[str]
    metadata: Optional[Dict[str, str]]


CreateBatchCompletionsV2Response: TypeAlias = BatchCompletionsJob


class UpdateBatchCompletionsV2Request(BaseModel):
    job_id: str = Field(description="ID of the batch completions job")
    priority: Optional[str] = Field(
        default=None,
        description="Priority of the batch inference job. Default to None.",
    )


class UpdateBatchCompletionsV2Response(BatchCompletionsJob):
    success: bool = Field(description="Whether the update was successful")


class CancelBatchCompletionsV2Request(BaseModel):
    job_id: str = Field(description="ID of the batch completions job")


class CancelBatchCompletionsV2Response(BaseModel):
    success: bool = Field(description="Whether the cancellation was successful")


class ListBatchCompletionV2Response(BaseModel):
    jobs: List[BatchCompletionsJob]


class GetBatchCompletionV2Response(BaseModel):
    job: BatchCompletionsJob
