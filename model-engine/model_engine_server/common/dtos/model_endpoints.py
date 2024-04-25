"""
Contains various input and output types relating to Model Bundles for the server.

TODO figure out how to do: (or if we want to do it)
List model endpoint history: GET model-endpoints/<endpoint id>/history
Read model endpoint creation logs: GET model-endpoints/<endpoint id>/creation-logs
"""

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from model_engine_server.domain.entities import (
    CallbackAuth,
    CpuSpecificationType,
    GpuType,
    ModelEndpointDeploymentState,
    ModelEndpointResourceState,
    ModelEndpointsSchema,
    ModelEndpointStatus,
    ModelEndpointType,
    StorageSpecificationType,
)
from pydantic import BaseModel, Field, HttpUrl


class BrokerType(str, Enum):
    """
    The list of available broker types for async endpoints.
    """

    REDIS = "redis"
    REDIS_24H = "redis_24h"
    SQS = "sqs"
    SERVICEBUS = "servicebus"


class BrokerName(str, Enum):
    """
    The list of available broker names for async endpoints.
    Broker name is only used in endpoint k8s annotations for the celery autoscaler.
    """

    REDIS = "redis-message-broker-master"
    SQS = "sqs-message-broker-master"
    SERVICEBUS = "servicebus-message-broker-master"


class CreateModelEndpointV1Request(BaseModel):
    name: str = Field(..., max_length=63)
    model_bundle_id: str
    endpoint_type: ModelEndpointType
    metadata: Dict[str, Any]  # TODO: JSON type
    post_inference_hooks: Optional[List[str]]
    cpus: CpuSpecificationType
    gpus: int = Field(..., ge=0)
    memory: StorageSpecificationType
    gpu_type: Optional[GpuType]
    storage: Optional[StorageSpecificationType]
    optimize_costs: Optional[bool]
    min_workers: int = Field(..., ge=0)
    max_workers: int = Field(..., ge=0)
    per_worker: int = Field(..., gt=0)
    labels: Dict[str, str]
    prewarm: Optional[bool]
    high_priority: Optional[bool]
    billing_tags: Optional[Dict[str, Any]]
    default_callback_url: Optional[HttpUrl]
    default_callback_auth: Optional[CallbackAuth]
    public_inference: Optional[bool] = Field(default=False)
    git_sha: Optional[str] = Field(default=None)
    disable_pod_rescheduling: Optional[bool]


class CreateModelEndpointV1Response(BaseModel):
    endpoint_creation_task_id: str


class UpdateModelEndpointV1Request(BaseModel):
    model_bundle_id: Optional[str]
    metadata: Optional[Dict[str, Any]]  # TODO: JSON type
    post_inference_hooks: Optional[List[str]]
    cpus: Optional[CpuSpecificationType]
    gpus: Optional[int] = Field(default=None, ge=0)
    memory: Optional[StorageSpecificationType]
    gpu_type: Optional[GpuType]
    storage: Optional[StorageSpecificationType]
    optimize_costs: Optional[bool]
    min_workers: Optional[int] = Field(default=None, ge=0)
    max_workers: Optional[int] = Field(default=None, ge=0)
    per_worker: Optional[int] = Field(default=None, gt=0)
    labels: Optional[Dict[str, str]]
    prewarm: Optional[bool]
    high_priority: Optional[bool]
    billing_tags: Optional[Dict[str, Any]]
    default_callback_url: Optional[HttpUrl]
    default_callback_auth: Optional[CallbackAuth]
    public_inference: Optional[bool]
    git_sha: Optional[str]
    disable_pod_rescheduling: Optional[bool]


class UpdateModelEndpointV1Response(BaseModel):
    endpoint_creation_task_id: str


class GetModelEndpointV1Response(BaseModel):
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
    git_sha: Optional[str] = Field(default=None)


class ListModelEndpointsV1Response(BaseModel):
    model_endpoints: List[GetModelEndpointV1Response]


class DeleteModelEndpointV1Response(BaseModel):
    deleted: bool


class ModelEndpointOrderBy(str, Enum):
    """
    The canonical list of possible orderings of Model Bundles.
    """

    NEWEST = "newest"
    OLDEST = "oldest"
    ALPHABETICAL = "alphabetical"


class GetModelEndpointsSchemaV1Response(BaseModel):
    model_endpoints_schema: ModelEndpointsSchema


# TODO history + creation logs
