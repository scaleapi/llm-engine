import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fastapi.openapi.models import OpenAPI
from llm_engine_server.common import dict_not_none
from llm_engine_server.common.serialization_utils import b64_to_python_json, python_json_to_b64
from llm_engine_server.domain.entities.common_types import (
    CpuSpecificationType,
    StorageSpecificationType,
)
from llm_engine_server.domain.entities.gpu_type import GpuType
from llm_engine_server.domain.entities.model_bundle_entity import ModelBundle
from llm_engine_server.domain.entities.owned_entity import OwnedEntity
from pydantic import BaseModel, Field
from typing_extensions import Literal

ModelEndpointsSchema = OpenAPI


class ModelEndpointType(str, Enum):
    ASYNC = "async"
    SYNC = "sync"
    STREAMING = "streaming"


class ModelEndpointStatus(str, Enum):
    # Duplicates common/types::EndpointStatus, when refactor is done, delete the old type
    # See EndpointStatus for status explanations
    READY = "READY"
    UPDATE_PENDING = "UPDATE_PENDING"
    UPDATE_IN_PROGRESS = "UPDATE_IN_PROGRESS"
    UPDATE_FAILED = "UPDATE_FAILED"
    DELETE_IN_PROGRESS = "DELETE_IN_PROGRESS"


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


class ModelEndpointDeploymentState(BaseModel):
    """
    This is the entity-layer class for the deployment settings related to a Model Endpoint.
    """

    min_workers: int = Field(..., ge=0)
    max_workers: int = Field(..., ge=0)
    per_worker: int = Field(..., gt=0)
    available_workers: Optional[int] = Field(default=None, ge=0)
    unavailable_workers: Optional[int] = Field(default=None, ge=0)


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


class ModelEndpointConfig(BaseModel):
    """
    This is the entity-layer class for the configuration of a Model Endpoint.
    """

    endpoint_name: str
    bundle_name: str
    post_inference_hooks: Optional[List[str]]
    user_id: Optional[str] = None
    default_callback_url: Optional[str] = None
    default_callback_auth: Optional[CallbackAuth]

    def serialize(self) -> str:
        return python_json_to_b64(dict_not_none(**self.dict()))

    @staticmethod
    def deserialize(serialized_config: str) -> "ModelEndpointConfig":
        return ModelEndpointConfig.parse_obj(b64_to_python_json(serialized_config))


class ModelEndpointUserConfigState(BaseModel):
    app_config: Optional[Dict[str, Any]]
    endpoint_config: Optional[ModelEndpointConfig]


class ModelEndpointRecord(OwnedEntity):
    """
    This is the entity-layer class for the record related to a Model Endpoint.

    TODO:  once we implement k8s caching, we should also add `labels` and `post_inference_hooks`.
    """

    id: str
    name: str
    created_by: str
    created_at: datetime.datetime
    last_updated_at: Optional[datetime.datetime]
    metadata: Optional[Dict[str, Any]]
    creation_task_id: Optional[str] = Field(default=None)
    endpoint_type: ModelEndpointType
    destination: str
    status: ModelEndpointStatus
    current_model_bundle: ModelBundle
    owner: str
    public_inference: Optional[bool]


class ModelEndpointInfraState(BaseModel):
    """
    This is the entity-layer class for the infrastructure state related to a Model Endpoint.
    """

    deployment_name: str
    aws_role: str
    results_s3_bucket: str
    child_fn_info: Optional[Dict[str, Any]]
    labels: Dict[str, str]
    deployment_state: ModelEndpointDeploymentState
    resource_state: ModelEndpointResourceState
    user_config_state: ModelEndpointUserConfigState
    prewarm: Optional[bool] = None
    high_priority: Optional[bool]
    num_queued_items: Optional[int]
    image: str


class ModelEndpoint(BaseModel):
    """
    This is the entity-layer class for everything related to a Model Endpoint.
    """

    record: ModelEndpointRecord
    infra_state: Optional[ModelEndpointInfraState]
