from enum import Enum
from typing import Literal, Union

from .pydantic_types import BaseModel, Field

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
