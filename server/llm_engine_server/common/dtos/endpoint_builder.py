from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from llm_engine_server.domain.entities import (
    CallbackAuth,
    CpuSpecificationType,
    GpuType,
    ModelEndpointRecord,
    StorageSpecificationType,
)


class BuildEndpointRequest(BaseModel):
    model_endpoint_record: ModelEndpointRecord
    deployment_name: str
    min_workers: int
    max_workers: int
    per_worker: int
    cpus: CpuSpecificationType
    gpus: int
    memory: StorageSpecificationType
    gpu_type: Optional[GpuType]
    storage: Optional[StorageSpecificationType]
    optimize_costs: bool
    aws_role: str
    results_s3_bucket: str
    child_fn_info: Optional[Dict[str, Any]]  # TODO: remove this if we don't need it.
    post_inference_hooks: Optional[List[str]]
    labels: Dict[str, str]
    prewarm: bool = True
    high_priority: Optional[bool]
    default_callback_url: Optional[str]
    default_callback_auth: Optional[CallbackAuth]


class BuildEndpointStatus(str, Enum):
    ENDPOINT_DELETED = "ENDPOINT_DELETED"
    OK = "OK"


class BuildEndpointResponse(BaseModel):
    status: BuildEndpointStatus
