from enum import Enum
from typing import Any, Dict, List, Optional

from model_engine_server.domain.entities import (
    CallbackAuth,
    CpuSpecificationType,
    GpuType,
    ModelEndpointRecord,
    StorageSpecificationType,
)
from pydantic import BaseModel


class BuildEndpointRequest(BaseModel):  # TODO multinode?
    model_endpoint_record: ModelEndpointRecord
    deployment_name: str
    min_workers: int
    max_workers: int
    per_worker: int
    cpus: CpuSpecificationType
    gpus: int
    memory: StorageSpecificationType
    gpu_type: Optional[GpuType] = None
    storage: Optional[StorageSpecificationType] = None
    optimize_costs: bool
    aws_role: str
    results_s3_bucket: str
    child_fn_info: Optional[Dict[str, Any]] = None  # TODO: remove this if we don't need it.
    post_inference_hooks: Optional[List[str]] = None
    labels: Dict[str, str]
    billing_tags: Optional[Dict[str, Any]] = None
    prewarm: bool = True
    high_priority: Optional[bool] = None
    default_callback_url: Optional[str] = None
    default_callback_auth: Optional[CallbackAuth] = None


class BuildEndpointStatus(str, Enum):
    ENDPOINT_DELETED = "ENDPOINT_DELETED"
    OK = "OK"


class BuildEndpointResponse(BaseModel):
    status: BuildEndpointStatus
