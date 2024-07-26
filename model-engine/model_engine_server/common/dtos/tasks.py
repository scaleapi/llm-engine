"""
DTOs for the task abstraction.
"""

from enum import Enum
from typing import Any, Optional

from model_engine_server.common.pydantic_types import BaseModel, Field, RootModel
from model_engine_server.domain.entities import CallbackAuth


class ResponseSchema(RootModel):
    root: Any


class RequestSchema(RootModel):
    root: Any


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNDEFINED = "UNDEFINED"


class CreateAsyncTaskV1Response(BaseModel):
    task_id: str


class GetAsyncTaskV1Response(BaseModel):
    task_id: str
    status: TaskStatus
    result: Optional[ResponseSchema] = None
    traceback: Optional[str] = None


class SyncEndpointPredictV1Response(BaseModel):
    status: TaskStatus
    result: Optional[Any] = None
    traceback: Optional[str] = None


class EndpointPredictV1Request(BaseModel):
    url: Optional[str] = None
    args: Optional[RequestSchema] = None
    cloudpickle: Optional[str] = None
    callback_url: Optional[str] = None
    callback_auth: Optional[CallbackAuth] = None
    return_pickled: bool = False


class SyncEndpointPredictV1Request(EndpointPredictV1Request):
    timeout_seconds: Optional[float] = Field(default=None, gt=0)
    num_retries: Optional[int] = Field(default=None, ge=0)
    # See live_{sync,streaming}_model_endpoint_inference_gateway to see how timeout_seconds/num_retries interact.
    # Also these fields are only relevant for sync endpoints
