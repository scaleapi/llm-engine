from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from spellbook_serve.domain.entities.model_bundle_entity import ModelBundle
from spellbook_serve.domain.entities.model_endpoint_entity import ModelEndpoint
from spellbook_serve.domain.entities.owned_entity import OwnedEntity


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
    completed_at: Optional[datetime]
    status: BatchJobStatus
    created_by: str
    owner: str
    model_bundle: ModelBundle
    model_endpoint_id: Optional[str]
    task_ids_location: Optional[str]
    result_location: Optional[str]


class BatchJobProgress(BaseModel):
    num_tasks_pending: Optional[int]
    num_tasks_completed: Optional[int]


class BatchJob(BaseModel):
    record: BatchJobRecord
    model_endpoint: Optional[ModelEndpoint]
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
    completed_at: Optional[datetime]
    status: BatchJobStatus  # the status map relatively nicely onto BatchJobStatus
