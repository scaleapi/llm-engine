from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from dataclasses_json import Undefined, dataclass_json


class EndpointType(Enum):
    sync_endpoint = "sync"
    async_endpoint = "async"

    @classmethod
    def deserialize(cls, serialized: str) -> "EndpointType":
        return EndpointType(serialized)

    @classmethod
    def serialize(cls, deserialized: "EndpointType") -> str:
        return deserialized.value


class EndpointStatus(Enum):
    """
    List of possible values for the endpoint_status column in the db
    This column pretty much stores what ongoing operations there are on the endpoint's k8s resources to prevent races.

    The transitions are as follows:
    On endpoint create:
        Gateway takes status from None to `update_pending`. Celery task takes status from `update_pending` to
        `update_in_progress` and then to `ready`, or `update_failed` if resource update has failed.
    On endpoint edit:
        Gateway takes status from `ready` to `update_pending`. Celery task takes status from `update_pending` to
        `update_in_progress` and then to `ready`, or `update_failed` if resource update has failed.
    On endpoint delete:
        Gateway takes status from `ready`, `update_pending` to `delete_in_progress`, then to None.
        If state is `update_in_progress`, deletion fails as of now. Later on we may move deletion to an async task.
    """

    ready = "READY"
    """
    Meaning: The db entry and all k8s resources are created, and the endpoint is ready to serve requests, other than
        pods needing to spin up. The actual status returned to the user needs to be more granular than this
    """

    update_pending = "UPDATE_PENDING"
    """
    Meaning: The db entry has just been created/edited,
        and an async task to create/edit the k8s resources is in progress.
    """

    update_in_progress = "UPDATE_IN_PROGRESS"
    """
    Meaning: The async task to create/edit k8s resources is about to make these resource changes.
        This state is distinct from the other update_pending state so that we can support deletions
        across a larger range of time.
    """

    update_failed = "UPDATE_FAILED"
    """
    Meaning: If the k8s resource editing has failed for some reason, we mark the state as such. We shouldn't
        encounter this unless there is a bug.
    """

    delete_in_progress = "DELETE_IN_PROGRESS"
    """
    Meaning: If the k8s resources and db entry are to be deleted soon. Currently just an informative status for users.
        In the future this may be necessary if we move deletion to an async task.
    """


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class EndpointParams:
    """
    A helper class that represents tweakable parameters on a ModelEndpoint, specifically on the
    k8s resources. This is similar to what gets passed to the endpoint creation celery task, but
    isn't exactly the same.
    """

    bundle_url: str
    image: Optional[str]  # Not present in celery request
    cpus: str
    gpus: int
    gpu_type: Optional[str]
    memory: str
    min_workers: int
    max_workers: int
    per_worker: int
    aws_role: str
    results_s3_bucket: str
    labels: Optional[Dict[str, str]]
    storage: Optional[str]
    prewarm: Optional[bool]
    high_priority: Optional[bool]

    def has_gpu_request(self):
        return int(self.gpus) > 0


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class EndpointBuilderParams(EndpointParams):
    """
    A helper class that materialize celery builder request.
    """

    endpoint_id: str
    endpoint_name: str
    env_params: Dict[str, Any]
    requirements: List[str]
    user_id: str  # TODO: Not sure if this is right - should be created_by?
    endpoint_type: str
    packaging_type: str
    deployment_name: str
    destination: str
    bundle_name: str
    bundle_metadata: Optional[Dict[str, Any]] = None
    app_config: Optional[Dict[str, Any]] = None
    child_fn_info: Optional[Dict[str, Any]] = None
    post_inference_hooks: Optional[List[str]] = None
