from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from model_engine_server.domain.entities.model_bundle_entity import ModelBundle
from model_engine_server.domain.entities.model_endpoint_entity import ModelEndpoint
from model_engine_server.domain.entities.owned_entity import OwnedEntity
from pydantic import BaseModel


class BatchJobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELLED = "CANCELLED"
    UNDEFINED = "UNDEFINED"
    TIMEOUT = "TIMEOUT"


class BatchJobSerializationFormat(str, Enum):
    JSON = "JSON"
    PICKLE = "PICKLE"


class BatchJobRecord(OwnedEntity):
    id: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    status: BatchJobStatus
    created_by: str
    owner: str
    model_bundle: ModelBundle
    model_endpoint_id: Optional[str] = None
    task_ids_location: Optional[str] = None
    result_location: Optional[str] = None


class BatchJobProgress(BaseModel):
    num_tasks_pending: Optional[int] = None
    num_tasks_completed: Optional[int] = None


class BatchJob(BaseModel):
    record: BatchJobRecord
    model_endpoint: Optional[ModelEndpoint] = None
    progress: BatchJobProgress


class DockerImageBatchJob(BaseModel):
    """
    This is the entity-layer class for a Docker Image Batch Job, i.e. a batch job
    created via the "supply a docker image for a k8s job" API.
    """

    id: str
    created_by: str
    owner: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    status: BatchJobStatus  # the status map relatively nicely onto BatchJobStatus
    annotations: Optional[Dict[str, str]] = None
    override_job_max_runtime_s: Optional[int] = None
    num_workers: Optional[int] = 1
