"""
DTOs for LLM APIs.
"""

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .core import (
    CallbackAuth,
    CpuSpecificationType,
    GpuType,
    ModelEndpointStatus,
    ModelEndpointType,
    StorageSpecificationType,
)
from .pydantic_types import BaseModel, Field, HttpUrl


class ModelEndpointDeploymentState(BaseModel):
    """
    This is the entity-layer class for the deployment settings related to a Model Endpoint.
    """

    min_workers: int = Field(..., ge=0)
    max_workers: int = Field(..., ge=0)
    per_worker: int = Field(..., gt=0)
    concurrent_requests_per_worker: int = Field(..., gt=0)
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
