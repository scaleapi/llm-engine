"""
DTOs for the batch job abstraction.
"""

from datetime import datetime, timedelta
from typing import Any, Collection, Dict, List, Optional

from model_engine_server.common import dict_not_none
from model_engine_server.common.pydantic_types import BaseModel, ConfigDict, model_validator
from model_engine_server.domain.entities import (
    BatchJobSerializationFormat,
    BatchJobStatus,
    CpuSpecificationType,
    DockerImageBatchJob,
    GpuType,
    StorageSpecificationType,
)


class CreateBatchJobResourceRequests(BaseModel):
    cpus: Optional[CpuSpecificationType] = None
    memory: Optional[StorageSpecificationType] = None
    gpus: Optional[int] = None
    gpu_type: Optional[GpuType] = None
    storage: Optional[StorageSpecificationType] = None
    max_workers: Optional[int] = None
    per_worker: Optional[int] = None


class CreateBatchJobV1Request(BaseModel):
    model_bundle_id: str
    input_path: str
    serialization_format: BatchJobSerializationFormat
    labels: Dict[str, str]
    resource_requests: CreateBatchJobResourceRequests
    timeout_seconds: float = 43200.0  # 12 hours


class CreateBatchJobV1Response(BaseModel):
    job_id: str


class GetBatchJobV1Response(BaseModel):
    status: BatchJobStatus
    result: Optional[str] = None
    duration: timedelta
    num_tasks_pending: Optional[int] = None
    num_tasks_completed: Optional[int] = None


class UpdateBatchJobV1Request(BaseModel):
    cancel: bool


class UpdateBatchJobV1Response(BaseModel):
    success: bool


# Docker Image Batch Job operations


class CreateDockerImageBatchJobResourceRequests(BaseModel):
    cpus: Optional[CpuSpecificationType] = None
    memory: Optional[StorageSpecificationType] = None
    gpus: Optional[int] = None
    gpu_type: Optional[GpuType] = None
    storage: Optional[StorageSpecificationType] = None
    nodes_per_worker: Optional[int] = (
        None  # TODO this is used only for inferring hardware, if multinode batch jobs is added we can reuse this field
    )
    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def merge_requests(
        cls,
        default_requests: "CreateDockerImageBatchJobResourceRequests",
        override_requests: "CreateDockerImageBatchJobResourceRequests",
    ) -> "CreateDockerImageBatchJobResourceRequests":
        default_dict = dict_not_none(**default_requests.dict())
        override_dict = dict_not_none(**override_requests.dict())
        default_dict.update(override_dict)
        return CreateDockerImageBatchJobResourceRequests.parse_obj(default_dict)

    @classmethod
    def common_requests(
        cls,
        requests_1: "CreateDockerImageBatchJobResourceRequests",
        requests_2: "CreateDockerImageBatchJobResourceRequests",
    ) -> Collection[str]:
        dict_1 = dict_not_none(**requests_1.dict())
        dict_2 = dict_not_none(**requests_2.dict())
        return set(dict_1.keys()).intersection(dict_2.keys())


class CreateDockerImageBatchJobV1Request(BaseModel):
    docker_image_batch_job_bundle_name: Optional[str] = None
    docker_image_batch_job_bundle_id: Optional[str] = None
    job_config: Optional[Dict[str, Any]] = None
    # TODO also expose a separate argument to pass an s3file to the job, as opposed to job_config
    labels: Dict[str, str]  # TODO this probably should go in the bundle

    resource_requests: CreateDockerImageBatchJobResourceRequests = (
        CreateDockerImageBatchJobResourceRequests()
    )

    override_job_max_runtime_s: Optional[int] = None

    @model_validator(mode="before")
    def exactly_one_name_or_id(cls, values):
        bundle_name = values.get("docker_image_batch_job_bundle_name")
        bundle_id = values.get("docker_image_batch_job_bundle_id")
        if bundle_name is None and bundle_id is None:
            raise ValueError(
                "At least one of docker_image_batch_job_bundle_name and docker_image_batch_job_bundle_id must be specified"
            )
        if bundle_name is not None and bundle_id is not None:
            raise ValueError(
                "Only one of docker_image_batch_job_bundle_name and docker_image_batch_job_bundle_id may be specified"
            )
        return values


class CreateDockerImageBatchJobV1Response(BaseModel):
    job_id: str


class GetDockerImageBatchJobV1Response(BaseModel):
    status: BatchJobStatus


class ListDockerImageBatchJobsV1Response(BaseModel):
    jobs: List[DockerImageBatchJob]


class UpdateDockerImageBatchJobV1Request(BaseModel):
    cancel: bool


class UpdateDockerImageBatchJobV1Response(BaseModel):
    success: bool


# Docker Image Batch Job Bundle operations


class CreateDockerImageBatchJobBundleV1Request(BaseModel):
    name: str
    image_repository: str
    image_tag: str
    command: List[str]
    env: Dict[str, str] = {}
    mount_location: Optional[str] = None
    resource_requests: CreateDockerImageBatchJobResourceRequests = (
        CreateDockerImageBatchJobResourceRequests()
    )
    public: Optional[bool] = False


class CreateDockerImageBatchJobBundleV1Response(BaseModel):
    docker_image_batch_job_bundle_id: str


class DockerImageBatchJobBundleV1Response(BaseModel):
    id: str
    name: str
    created_at: datetime
    image_repository: str
    image_tag: str
    command: List[str]
    env: Dict[str, str]
    mount_location: Optional[str] = None
    cpus: Optional[str] = None
    memory: Optional[str] = None
    storage: Optional[str] = None
    gpus: Optional[int] = None
    gpu_type: Optional[str] = None
    public: Optional[bool] = None
    model_config = ConfigDict(from_attributes=True)


class ListDockerImageBatchJobBundleV1Response(BaseModel):
    docker_image_batch_job_bundles: List[DockerImageBatchJobBundleV1Response]
