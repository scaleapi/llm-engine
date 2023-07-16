"""
DTOs for the task abstraction.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from llm_engine_server.domain.entities import CallbackAuth


class ResponseSchema(BaseModel):
    __root__: Any


class RequestSchema(BaseModel):
    __root__: Any


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
