import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, TypedDict, Union

from model_engine_server.common.config import hmi_config
from model_engine_server.common.dtos.model_endpoints import BrokerName, BrokerType
from model_engine_server.common.dtos.resource_manager import CreateOrUpdateResourcesRequest
from model_engine_server.common.env_vars import CIRCLECI, GIT_TAG
from model_engine_server.common.resource_limits import (
    FORWARDER_CPU_USAGE,
    FORWARDER_MEMORY_USAGE,
    FORWARDER_STORAGE_USAGE,
    FORWARDER_WORKER_COUNT,
)
from model_engine_server.common.serialization_utils import python_json_to_b64
from model_engine_server.core.config import infra_config
from model_engine_server.domain.entities import (
    ModelEndpointConfig,
    RunnableImageLike,
    StreamingEnhancedRunnableImageFlavor,
    TritonEnhancedRunnableImageFlavor,
)
from model_engine_server.domain.use_cases.model_endpoint_use_cases import (
    CONVERTED_FROM_ARTIFACT_LIKE_KEY,
)
from model_engine_server.infra.gateways.k8s_resource_parser import (
    get_node_port,
    get_target_concurrency_from_per_worker_value,
)

__all__: Sequence[str] = (
    "BatchJobOrchestrationJobArguments",
    "CommonEndpointParams",
    "DeploymentRunnableImageAsyncCpuArguments",
    "DeploymentRunnableImageAsyncGpuArguments",
    "DeploymentRunnableImageStreamingCpuArguments",
    "DeploymentRunnableImageStreamingGpuArguments",
    "DeploymentRunnableImageSyncCpuArguments",
    "DeploymentRunnableImageSyncGpuArguments",
    "DeploymentTritonEnhancedRunnableImageAsyncCpuArguments",
    "DeploymentTritonEnhancedRunnableImageAsyncGpuArguments",
    "DeploymentTritonEnhancedRunnableImageSyncCpuArguments",
    "DeploymentTritonEnhancedRunnableImageSyncGpuArguments",
    "DestinationRuleArguments",
    "DictStrInt",
    "DictStrStr",
    "DockerImageBatchJobCpuArguments",
    "DockerImageBatchJobGpuArguments",
    "EndpointConfigArguments",
    "EndpointResourceArguments",
    "HorizontalAutoscalingEndpointParams",
    "HorizontalPodAutoscalerArguments",
    "ImageCacheArguments",
    "CronTriggerArguments",
    "LAUNCH_DEFAULT_PRIORITY_CLASS",
    "LAUNCH_HIGH_PRIORITY_CLASS",
    "ResourceArguments",
    "ServiceArguments",
    "UserConfigArguments",
    "VerticalAutoscalingEndpointParams",
    "VerticalPodAutoscalerArguments",
    "VirtualServiceArguments",
    "get_endpoint_resource_arguments_from_request",
)

# Constants for Launch priority classes
LAUNCH_HIGH_PRIORITY_CLASS = "model-engine-high-priority"
LAUNCH_DEFAULT_PRIORITY_CLASS = "model-engine-default-priority"

IMAGE_HASH_MAX_LENGTH = 32
FORWARDER_PORT = 5000
USER_CONTAINER_PORT = 5005
ARTIFACT_LIKE_CONTAINER_PORT = FORWARDER_PORT


class _BaseResourceArguments(TypedDict):
    """Keyword-arguments for substituting into all resource templates."""

    RESOURCE_NAME: str
    NAMESPACE: str
    TEAM: str
    PRODUCT: str
    CREATED_BY: str
    OWNER: str
    GIT_TAG: str


class _BaseEndpointArguments(_BaseResourceArguments):
    """Keyword-arguments for substituting into all endpoint resource templates."""

    ENDPOINT_ID: str
    ENDPOINT_NAME: str


class DictStrStr(str):
    """A string that can be converted to a dict of str:str."""

    pass


class _BaseDeploymentArguments(_BaseEndpointArguments):
    """Keyword-arguments for substituting into all deployment templates (both sync and async)."""

    CHANGE_CAUSE_MESSAGE: str
    AWS_ROLE: str
    PRIORITY: str
    IMAGE: str
    IMAGE_HASH: str
    DD_TRACE_ENABLED: str
    CPUS: str
    MEMORY: str
    STORAGE_DICT: DictStrStr
    PER_WORKER: int
    MIN_WORKERS: int
    MAX_WORKERS: int
    RESULTS_S3_BUCKET: str


class _AsyncDeploymentArguments(TypedDict):
    """Keyword-arguments for substituting into async deployment templates."""

    CELERY_S3_BUCKET: str
    QUEUE: str
    BROKER_NAME: str
    BROKER_TYPE: str
    SQS_QUEUE_URL: str
    SQS_PROFILE: str


class _SyncArtifactDeploymentArguments(TypedDict):
    """Keyword-arguments for substituting into sync deployment templates."""

    ARTIFACT_LIKE_CONTAINER_PORT: int


class _SyncRunnableImageDeploymentArguments(TypedDict):
    """Keyword-arguments for substituting into sync deployment templates."""

    FORWARDER_PORT: int
    FORWARDER_WORKER_COUNT: int


class _StreamingDeploymentArguments(TypedDict):
    """Keyword-arguments for substituting into streaming deployment templates."""

    FORWARDER_PORT: int
    STREAMING_PREDICT_ROUTE: str
    FORWARDER_WORKER_COUNT: int


class _RunnableImageDeploymentArguments(_BaseDeploymentArguments):
    """Keyword-arguments for substituting into runnable image deployment templates."""

    MAIN_ENV: List[Dict[str, Any]]
    COMMAND: List[str]
    PREDICT_ROUTE: str
    HEALTHCHECK_ROUTE: str
    READINESS_INITIAL_DELAY: int
    INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH: str
    FORWARDER_CONFIG_FILE_NAME: str
    FORWARDER_CPUS_LIMIT: float
    FORWARDER_MEMORY_LIMIT: str
    FORWARDER_STORAGE_LIMIT: str
    USER_CONTAINER_PORT: int


class _JobArguments(_BaseResourceArguments):
    """Keyword-arguments for substituting into all job templates."""

    JOB_ID: str
    BATCH_JOB_MAX_RUNTIME: int
    BATCH_JOB_TTL_SECONDS_AFTER_FINISHED: int
    REQUEST_ID: str


class _DockerImageBatchJobArguments(_JobArguments):
    """Keyword-arguments for substituting into docker-image-batch-job templates."""

    AWS_ROLE: str
    IMAGE: str
    CPUS: str
    MEMORY: str
    STORAGE_DICT: DictStrStr
    MOUNT_PATH: str
    INPUT_LOCATION: str
    S3_FILE: str
    LOCAL_FILE_NAME: str
    FILE_CONTENTS_B64ENCODED: str
    COMMAND: List[str]
    BATCH_JOB_NUM_WORKERS: int


class _GpuArguments(TypedDict):
    """Keyword-arguments for substituting into gpu templates."""

    GPU_TYPE: str
    GPUS: int


class _TritonArguments(TypedDict):
    """Keyword-arguments for substituting into Triton templates."""

    TRITON_MODEL_REPOSITORY: str
    TRITON_READINESS_INITIAL_DELAY: int
    TRITON_CPUS: str
    TRITON_MEMORY_DICT: DictStrStr
    TRITON_STORAGE_DICT: DictStrStr
    TRITON_COMMAND: str
    TRITON_COMMIT_TAG: str


class DeploymentRunnableImageSyncCpuArguments(
    _RunnableImageDeploymentArguments, _SyncRunnableImageDeploymentArguments
):
    """Keyword-arguments for substituting into CPU sync deployment templates for runnable images."""


class DeploymentRunnableImageSyncGpuArguments(
    _RunnableImageDeploymentArguments, _SyncRunnableImageDeploymentArguments, _GpuArguments
):
    """Keyword-arguments for substituting into GPU sync deployment templates for runnable images."""


class DeploymentRunnableImageStreamingCpuArguments(
    _RunnableImageDeploymentArguments, _StreamingDeploymentArguments
):
    """Keyword-arguments for substituting into CPU streaming deployment templates for runnable images."""


class DeploymentRunnableImageStreamingGpuArguments(
    _RunnableImageDeploymentArguments, _StreamingDeploymentArguments, _GpuArguments
):
    """Keyword-arguments for substituting into GPU streaming deployment templates for runnable images."""


class DeploymentRunnableImageAsyncCpuArguments(
    _RunnableImageDeploymentArguments, _AsyncDeploymentArguments
):
    """Keyword-arguments for substituting CPU async deployment templates for runnable images."""


class DeploymentRunnableImageAsyncGpuArguments(
    _RunnableImageDeploymentArguments, _AsyncDeploymentArguments, _GpuArguments
):
    """Keyword-arguments for substituting GPU async deployment templates for runnable images."""


class DeploymentTritonEnhancedRunnableImageSyncCpuArguments(
    _RunnableImageDeploymentArguments, _SyncRunnableImageDeploymentArguments, _TritonArguments
):
    """Keyword-arguments for substituting into CPU sync deployment templates for triton-enhanced
    runnable images.
    """


class DeploymentTritonEnhancedRunnableImageSyncGpuArguments(
    _RunnableImageDeploymentArguments,
    _SyncRunnableImageDeploymentArguments,
    _GpuArguments,
    _TritonArguments,
):
    """Keyword-arguments for substituting into GPU sync deployment templates for triton-enhanced
    runnable images.
    """


class DeploymentTritonEnhancedRunnableImageAsyncCpuArguments(
    _RunnableImageDeploymentArguments, _AsyncDeploymentArguments, _TritonArguments
):
    """Keyword-arguments for substituting CPU async deployment templates for triton-enhanced
    runnable images.
    """


class DeploymentTritonEnhancedRunnableImageAsyncGpuArguments(
    _RunnableImageDeploymentArguments, _AsyncDeploymentArguments, _GpuArguments, _TritonArguments
):
    """Keyword-arguments for substituting GPU async deployment templates for triton-enhanced
    runnable images.
    """


class HorizontalPodAutoscalerArguments(_BaseEndpointArguments):
    """Keyword-arguments for substituting into horizontal pod autoscaler templates."""

    MIN_WORKERS: int
    MAX_WORKERS: int
    CONCURRENCY: float
    API_VERSION: str


class KedaScaledObjectArguments(_BaseEndpointArguments):
    MIN_WORKERS: int
    MAX_WORKERS: int
    # CONCURRENCY: float  # TODO add in when we scale from 1 -> N pods
    REDIS_HOST_PORT: str
    REDIS_DB_INDEX: str
    AUTHENTICATION_REF: str


class UserConfigArguments(_BaseEndpointArguments):
    """Keyword-arguments for substituting into user-config templates."""

    CONFIG_DATA_SERIALIZED: str


class EndpointConfigArguments(_BaseEndpointArguments):
    """Keyword-arguments for substituting into endpoint-config templates."""

    ENDPOINT_CONFIG_SERIALIZED: str


class DictStrInt(str):
    """A string that can be converted to a dict of str:int."""

    pass


class ServiceArguments(_BaseEndpointArguments):
    """Keyword-arguments for substituting into service templates."""

    SERVICE_TYPE: str
    SERVICE_TARGET_PORT: int
    NODE_PORT_DICT: DictStrInt


class DestinationRuleArguments(_BaseEndpointArguments):
    """Keyword-arguments for substituting into destination-rule templates."""


class VerticalPodAutoscalerArguments(_BaseEndpointArguments):
    """Keyword-arguments for substituting into vertical pod autoscaler templates."""

    CPUS: str
    MEMORY: str


class PodDisruptionBudgetArguments(_BaseEndpointArguments):
    """Keyword-arguments for substituting into pod disruption budget templates."""

    pass


class VirtualServiceArguments(_BaseEndpointArguments):
    """Keyword-arguments for substituting into virtual-service templates."""

    DNS_HOST_DOMAIN: str


class BatchJobOrchestrationJobArguments(_JobArguments):
    """Keyword-arguments for substituting into batch-job-orchestration-job templates."""

    SERIALIZATION_FORMAT: str
    INPUT_LOCATION: str
    BATCH_JOB_TIMEOUT: float


class DockerImageBatchJobCpuArguments(_DockerImageBatchJobArguments):
    """Keyword-arguments for substituting into docker-image-batch-job-cpu templates."""


class DockerImageBatchJobGpuArguments(_DockerImageBatchJobArguments, _GpuArguments):
    """Keyword-arguments for substituting into docker-image-batch-job-gpu templates."""


class ImageCacheArguments(TypedDict):
    """Keyword-arguments for substituting into image-cache templates."""

    RESOURCE_NAME: str
    NAMESPACE: str


class CronTriggerArguments(TypedDict):
    """Keyword-arguments for substituting into cronjob trigger templates."""

    HOST: str
    NAME: str
    CREATED_BY: str
    OWNER: str
    TEAM: str
    PRODUCT: str
    TRIGGER_ID: str
    CRON_SCHEDULE: str
    DOCKER_IMAGE_BATCH_JOB_BUNDLE_ID: str
    JOB_CONFIG: str
    JOB_METADATA: str
    BATCH_CURL_JOB_ACTIVE_DEADLINE_SECONDS: int


class CommonEndpointParams(TypedDict):
    cpus: str
    memory: str
    gpus: int
    gpu_type: Optional[str]
    storage: Optional[str]
    bundle_url: str
    aws_role: str
    results_s3_bucket: str
    image: str
    labels: Any


class HorizontalAutoscalingEndpointParams(TypedDict):
    min_workers: int
    max_workers: int
    per_worker: int


class VerticalAutoscalingEndpointParams(TypedDict):
    min_cpu: str
    max_cpu: str
    min_memory: str
    max_memory: str


EndpointResourceArguments = Union[
    DeploymentRunnableImageAsyncCpuArguments,
    DeploymentRunnableImageAsyncGpuArguments,
    DeploymentRunnableImageStreamingCpuArguments,
    DeploymentRunnableImageStreamingGpuArguments,
    DeploymentRunnableImageSyncCpuArguments,
    DeploymentRunnableImageSyncGpuArguments,
    DeploymentTritonEnhancedRunnableImageAsyncCpuArguments,
    DeploymentTritonEnhancedRunnableImageAsyncGpuArguments,
    DeploymentTritonEnhancedRunnableImageSyncCpuArguments,
    DeploymentTritonEnhancedRunnableImageSyncGpuArguments,
    DestinationRuleArguments,
    EndpointConfigArguments,
    HorizontalPodAutoscalerArguments,
    ServiceArguments,
    UserConfigArguments,
    VerticalPodAutoscalerArguments,
    VirtualServiceArguments,
]

ResourceArguments = Union[
    EndpointResourceArguments,
    BatchJobOrchestrationJobArguments,
    DockerImageBatchJobCpuArguments,
    DockerImageBatchJobGpuArguments,
    ImageCacheArguments,
    CronTriggerArguments,
]


def compute_image_hash(image: str) -> str:
    return str(hashlib.sha256(str(image).encode()).hexdigest())[:IMAGE_HASH_MAX_LENGTH]


def container_start_triton_cmd(
    triton_model_repository: str,
    triton_model_replicas: Union[Dict[str, int], Dict[str, str]],
    ipv6_healthcheck: bool = False,
) -> List[str]:
    # NOTE: this path is set in the Triton-specific Dockerfile:
    triton_start_command: List[str] = [
        "python",
        "/install/tritonserver.py",
        "--model-repository",
        triton_model_repository,
    ]
    if ipv6_healthcheck:
        triton_start_command.append("--ipv6-healthcheck")
    for model_name, replica_count in triton_model_replicas.items():
        triton_start_command.extend(["--model-replicas", f"{model_name}:{replica_count}"])
    return triton_start_command


def get_endpoint_resource_arguments_from_request(
    k8s_resource_group_name: str,
    request: CreateOrUpdateResourcesRequest,
    sqs_queue_name: str,
    sqs_queue_url: str,
    endpoint_resource_name: str,
    api_version: str = "",
) -> EndpointResourceArguments:
    """Get the arguments for the endpoint resource templates from the request.

    This method applies only to endpoint resources, not to batch job resources.
    """
    build_endpoint_request = request.build_endpoint_request
    model_endpoint_record = build_endpoint_request.model_endpoint_record
    model_bundle = model_endpoint_record.current_model_bundle
    flavor = model_bundle.flavor
    user_id = model_endpoint_record.owner
    created_by = model_endpoint_record.created_by
    owner = model_endpoint_record.owner
    k8s_labels = build_endpoint_request.labels or {}
    team = k8s_labels.get("team", "")
    product = k8s_labels.get("product", "")
    storage = build_endpoint_request.storage
    sqs_profile = f"eks-{infra_config().profile_ml_worker}"  # TODO: Make this configurable
    s3_bucket = infra_config().s3_bucket

    storage_dict = DictStrStr("")
    if storage is not None:
        storage_dict = DictStrStr(f'ephemeral-storage: "{storage}"')

    change_cause_message = (
        f"Deployment at {datetime.utcnow()} UTC. "
        f"Using deployment constructed from model bundle ID {model_bundle.id}, "
        f"model bundle name {model_bundle.name}, "
        f"endpoint ID {model_endpoint_record.id}"
    )

    priority = LAUNCH_DEFAULT_PRIORITY_CLASS
    if build_endpoint_request.high_priority:
        priority = LAUNCH_HIGH_PRIORITY_CLASS

    image_hash = compute_image_hash(request.image)

    # In Circle CI, we use Redis on localhost instead of SQS
    if CIRCLECI:
        broker_name = BrokerName.REDIS.value
        broker_type = BrokerType.REDIS.value
    elif infra_config().cloud_provider == "azure":
        broker_name = BrokerName.SERVICEBUS.value
        broker_type = BrokerType.SERVICEBUS.value
    else:
        broker_name = BrokerName.SQS.value
        broker_type = BrokerType.SQS.value
    dd_trace_enabled = hmi_config.dd_trace_enabled
    if broker_type != BrokerType.SQS.value:
        sqs_queue_url = ""

    main_env = []
    if isinstance(flavor, RunnableImageLike) and flavor.env:
        main_env = [{"name": key, "value": value} for key, value in flavor.env.items()]
    main_env.append({"name": "AWS_PROFILE", "value": build_endpoint_request.aws_role})
    # NOTE: /opt/.aws/config is where service_template_config_map.yaml mounts the AWS config file, point to the mount for boto clients
    main_env.append({"name": "AWS_CONFIG_FILE", "value": "/opt/.aws/config"})

    infra_service_config_volume_mount_path = "/infra-config"
    forwarder_config_file_name = "service--forwarder.yaml"
    if (
        isinstance(flavor, RunnableImageLike)
        and request.build_endpoint_request.model_endpoint_record.metadata is not None
        and CONVERTED_FROM_ARTIFACT_LIKE_KEY
        in request.build_endpoint_request.model_endpoint_record.metadata
        and request.build_endpoint_request.model_endpoint_record.metadata[
            CONVERTED_FROM_ARTIFACT_LIKE_KEY
        ]
    ):
        if not flavor.env or "BASE_PATH" not in flavor.env:
            raise ValueError(
                "flavor.env['BASE_PATH'] is required for runnable image converted from artifact like bundle"
            )
        infra_service_config_volume_mount_path = (
            f"{flavor.env['BASE_PATH']}/model-engine/model_engine_server/core/configs"
        )
        forwarder_config_file_name = "service--forwarder-runnable-img-converted-from-artifact.yaml"

    triton_command = ""
    triton_memory = DictStrStr("")
    triton_storage = DictStrStr("")
    if isinstance(flavor, TritonEnhancedRunnableImageFlavor):
        triton_start_command = container_start_triton_cmd(
            flavor.triton_model_repository,
            flavor.triton_model_replicas or {},
            ipv6_healthcheck=True,
        )
        triton_command = " ".join(triton_start_command)
        if flavor.triton_memory is not None:
            triton_memory = DictStrStr(f'memory: "{flavor.triton_memory}"')
        if flavor.triton_storage is not None:
            triton_storage = DictStrStr(f'ephemeral-storage: "{flavor.triton_storage}"')

    if endpoint_resource_name == "deployment-runnable-image-async-cpu":
        assert isinstance(flavor, RunnableImageLike)
        return DeploymentRunnableImageAsyncCpuArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DD_TRACE_ENABLED=dd_trace_enabled,
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
            STORAGE_DICT=storage_dict,
            PER_WORKER=build_endpoint_request.per_worker,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
            RESULTS_S3_BUCKET=s3_bucket,
            # Runnable Image Arguments
            MAIN_ENV=main_env,
            COMMAND=flavor.command,
            PREDICT_ROUTE=flavor.predict_route,
            HEALTHCHECK_ROUTE=flavor.healthcheck_route,
            READINESS_INITIAL_DELAY=flavor.readiness_initial_delay_seconds,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Async Deployment Arguments
            CELERY_S3_BUCKET=s3_bucket,
            QUEUE=sqs_queue_name,
            BROKER_NAME=broker_name,
            BROKER_TYPE=broker_type,
            SQS_QUEUE_URL=sqs_queue_url,
            SQS_PROFILE=sqs_profile,
        )
    elif endpoint_resource_name == "deployment-runnable-image-async-gpu":
        assert isinstance(flavor, RunnableImageLike)
        assert build_endpoint_request.gpu_type is not None
        return DeploymentRunnableImageAsyncGpuArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DD_TRACE_ENABLED=dd_trace_enabled,
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
            STORAGE_DICT=storage_dict,
            PER_WORKER=build_endpoint_request.per_worker,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
            RESULTS_S3_BUCKET=s3_bucket,
            # Runnable Image Arguments
            MAIN_ENV=main_env,
            COMMAND=flavor.command,
            PREDICT_ROUTE=flavor.predict_route,
            HEALTHCHECK_ROUTE=flavor.healthcheck_route,
            READINESS_INITIAL_DELAY=flavor.readiness_initial_delay_seconds,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Async Deployment Arguments
            CELERY_S3_BUCKET=s3_bucket,
            QUEUE=sqs_queue_name,
            BROKER_NAME=broker_name,
            BROKER_TYPE=broker_type,
            SQS_QUEUE_URL=sqs_queue_url,
            SQS_PROFILE=sqs_profile,
            # GPU Deployment Arguments
            GPU_TYPE=build_endpoint_request.gpu_type.value,
            GPUS=build_endpoint_request.gpus,
        )
    elif endpoint_resource_name == "deployment-runnable-image-streaming-cpu":
        assert isinstance(flavor, StreamingEnhancedRunnableImageFlavor)
        return DeploymentRunnableImageStreamingCpuArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DD_TRACE_ENABLED=dd_trace_enabled,
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
            STORAGE_DICT=storage_dict,
            PER_WORKER=build_endpoint_request.per_worker,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
            RESULTS_S3_BUCKET=s3_bucket,
            # Runnable Image Arguments
            MAIN_ENV=main_env,
            COMMAND=flavor.streaming_command,
            PREDICT_ROUTE=flavor.predict_route,
            STREAMING_PREDICT_ROUTE=flavor.streaming_predict_route,
            HEALTHCHECK_ROUTE=flavor.healthcheck_route,
            READINESS_INITIAL_DELAY=flavor.readiness_initial_delay_seconds,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Streaming Deployment Arguments
            FORWARDER_PORT=FORWARDER_PORT,
            FORWARDER_WORKER_COUNT=FORWARDER_WORKER_COUNT,
        )
    elif endpoint_resource_name == "deployment-runnable-image-streaming-gpu":
        assert isinstance(flavor, StreamingEnhancedRunnableImageFlavor)
        assert build_endpoint_request.gpu_type is not None
        return DeploymentRunnableImageStreamingGpuArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DD_TRACE_ENABLED=dd_trace_enabled,
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
            STORAGE_DICT=storage_dict,
            PER_WORKER=build_endpoint_request.per_worker,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
            RESULTS_S3_BUCKET=s3_bucket,
            # Runnable Image Arguments
            MAIN_ENV=main_env,
            COMMAND=flavor.streaming_command,
            PREDICT_ROUTE=flavor.predict_route,
            STREAMING_PREDICT_ROUTE=flavor.streaming_predict_route,
            HEALTHCHECK_ROUTE=flavor.healthcheck_route,
            READINESS_INITIAL_DELAY=flavor.readiness_initial_delay_seconds,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Streaming Deployment Arguments
            FORWARDER_PORT=FORWARDER_PORT,
            FORWARDER_WORKER_COUNT=FORWARDER_WORKER_COUNT,
            # GPU Deployment Arguments
            GPU_TYPE=build_endpoint_request.gpu_type.value,
            GPUS=build_endpoint_request.gpus,
        )
    elif endpoint_resource_name == "deployment-runnable-image-sync-cpu":
        assert isinstance(flavor, RunnableImageLike)
        return DeploymentRunnableImageSyncCpuArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DD_TRACE_ENABLED=dd_trace_enabled,
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
            STORAGE_DICT=storage_dict,
            PER_WORKER=build_endpoint_request.per_worker,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
            RESULTS_S3_BUCKET=s3_bucket,
            # Runnable Image Arguments
            MAIN_ENV=main_env,
            COMMAND=flavor.command,
            PREDICT_ROUTE=flavor.predict_route,
            HEALTHCHECK_ROUTE=flavor.healthcheck_route,
            READINESS_INITIAL_DELAY=flavor.readiness_initial_delay_seconds,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Sync Deployment Arguments
            FORWARDER_PORT=FORWARDER_PORT,
            FORWARDER_WORKER_COUNT=FORWARDER_WORKER_COUNT,
        )
    elif endpoint_resource_name == "deployment-runnable-image-sync-gpu":
        assert isinstance(flavor, RunnableImageLike)
        assert build_endpoint_request.gpu_type is not None
        return DeploymentRunnableImageSyncGpuArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DD_TRACE_ENABLED=dd_trace_enabled,
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
            STORAGE_DICT=storage_dict,
            PER_WORKER=build_endpoint_request.per_worker,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
            RESULTS_S3_BUCKET=s3_bucket,
            # Runnable Image Arguments
            MAIN_ENV=main_env,
            COMMAND=flavor.command,
            PREDICT_ROUTE=flavor.predict_route,
            HEALTHCHECK_ROUTE=flavor.healthcheck_route,
            READINESS_INITIAL_DELAY=flavor.readiness_initial_delay_seconds,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Sync Deployment Arguments
            FORWARDER_PORT=FORWARDER_PORT,
            FORWARDER_WORKER_COUNT=FORWARDER_WORKER_COUNT,
            # GPU Deployment Arguments
            GPU_TYPE=build_endpoint_request.gpu_type.value,
            GPUS=build_endpoint_request.gpus,
        )
    elif endpoint_resource_name == "deployment-triton-enhanced-runnable-image-async-cpu":
        assert isinstance(flavor, TritonEnhancedRunnableImageFlavor)
        return DeploymentTritonEnhancedRunnableImageAsyncCpuArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DD_TRACE_ENABLED=dd_trace_enabled,
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
            STORAGE_DICT=storage_dict,
            PER_WORKER=build_endpoint_request.per_worker,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
            RESULTS_S3_BUCKET=s3_bucket,
            # Runnable Image Arguments
            MAIN_ENV=main_env,
            COMMAND=flavor.command,
            PREDICT_ROUTE=flavor.predict_route,
            HEALTHCHECK_ROUTE=flavor.healthcheck_route,
            READINESS_INITIAL_DELAY=flavor.readiness_initial_delay_seconds,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Async Deployment Arguments
            CELERY_S3_BUCKET=s3_bucket,
            QUEUE=sqs_queue_name,
            BROKER_NAME=broker_name,
            BROKER_TYPE=broker_type,
            SQS_QUEUE_URL=sqs_queue_url,
            SQS_PROFILE=sqs_profile,
            # Triton Deployment Arguments
            TRITON_MODEL_REPOSITORY=flavor.triton_model_repository,
            TRITON_CPUS=str(flavor.triton_num_cpu),
            TRITON_MEMORY_DICT=triton_memory,
            TRITON_STORAGE_DICT=triton_storage,
            TRITON_READINESS_INITIAL_DELAY=flavor.triton_readiness_initial_delay_seconds,
            TRITON_COMMAND=triton_command,
            TRITON_COMMIT_TAG=flavor.triton_commit_tag,
        )
    elif endpoint_resource_name == "deployment-triton-enhanced-runnable-image-async-gpu":
        assert isinstance(flavor, TritonEnhancedRunnableImageFlavor)
        assert build_endpoint_request.gpu_type is not None
        return DeploymentTritonEnhancedRunnableImageAsyncGpuArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DD_TRACE_ENABLED=dd_trace_enabled,
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
            STORAGE_DICT=storage_dict,
            PER_WORKER=build_endpoint_request.per_worker,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
            RESULTS_S3_BUCKET=s3_bucket,
            # Runnable Image Arguments
            MAIN_ENV=main_env,
            COMMAND=flavor.command,
            PREDICT_ROUTE=flavor.predict_route,
            HEALTHCHECK_ROUTE=flavor.healthcheck_route,
            READINESS_INITIAL_DELAY=flavor.readiness_initial_delay_seconds,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Async Deployment Arguments
            CELERY_S3_BUCKET=s3_bucket,
            QUEUE=sqs_queue_name,
            BROKER_NAME=broker_name,
            BROKER_TYPE=broker_type,
            SQS_QUEUE_URL=sqs_queue_url,
            SQS_PROFILE=sqs_profile,
            # GPU Deployment Arguments
            GPU_TYPE=build_endpoint_request.gpu_type.value,
            GPUS=build_endpoint_request.gpus,
            # Triton Deployment Arguments
            TRITON_MODEL_REPOSITORY=flavor.triton_model_repository,
            TRITON_CPUS=str(flavor.triton_num_cpu),
            TRITON_MEMORY_DICT=triton_memory,
            TRITON_STORAGE_DICT=triton_storage,
            TRITON_READINESS_INITIAL_DELAY=flavor.triton_readiness_initial_delay_seconds,
            TRITON_COMMAND=triton_command,
            TRITON_COMMIT_TAG=flavor.triton_commit_tag,
        )
    elif endpoint_resource_name == "deployment-triton-enhanced-runnable-image-sync-cpu":
        assert isinstance(flavor, TritonEnhancedRunnableImageFlavor)
        return DeploymentTritonEnhancedRunnableImageSyncCpuArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DD_TRACE_ENABLED=dd_trace_enabled,
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
            STORAGE_DICT=storage_dict,
            PER_WORKER=build_endpoint_request.per_worker,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
            RESULTS_S3_BUCKET=s3_bucket,
            # Runnable Image Arguments
            MAIN_ENV=main_env,
            COMMAND=flavor.command,
            PREDICT_ROUTE=flavor.predict_route,
            HEALTHCHECK_ROUTE=flavor.healthcheck_route,
            READINESS_INITIAL_DELAY=flavor.readiness_initial_delay_seconds,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Sync Deployment Arguments
            FORWARDER_PORT=FORWARDER_PORT,
            FORWARDER_WORKER_COUNT=FORWARDER_WORKER_COUNT,
            # Triton Deployment Arguments
            TRITON_MODEL_REPOSITORY=flavor.triton_model_repository,
            TRITON_CPUS=str(flavor.triton_num_cpu),
            TRITON_MEMORY_DICT=triton_memory,
            TRITON_STORAGE_DICT=triton_storage,
            TRITON_READINESS_INITIAL_DELAY=flavor.triton_readiness_initial_delay_seconds,
            TRITON_COMMAND=triton_command,
            TRITON_COMMIT_TAG=flavor.triton_commit_tag,
        )
    elif endpoint_resource_name == "deployment-triton-enhanced-runnable-image-sync-gpu":
        assert isinstance(flavor, TritonEnhancedRunnableImageFlavor)
        assert build_endpoint_request.gpu_type is not None
        return DeploymentTritonEnhancedRunnableImageSyncGpuArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DD_TRACE_ENABLED=dd_trace_enabled,
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
            STORAGE_DICT=storage_dict,
            PER_WORKER=build_endpoint_request.per_worker,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
            RESULTS_S3_BUCKET=s3_bucket,
            # Runnable Image Arguments
            MAIN_ENV=main_env,
            COMMAND=flavor.command,
            PREDICT_ROUTE=flavor.predict_route,
            HEALTHCHECK_ROUTE=flavor.healthcheck_route,
            READINESS_INITIAL_DELAY=flavor.readiness_initial_delay_seconds,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Sync Deployment Arguments
            FORWARDER_PORT=FORWARDER_PORT,
            FORWARDER_WORKER_COUNT=FORWARDER_WORKER_COUNT,
            # GPU Deployment Arguments
            GPU_TYPE=build_endpoint_request.gpu_type.value,
            GPUS=build_endpoint_request.gpus,
            # Triton Deployment Arguments
            TRITON_MODEL_REPOSITORY=flavor.triton_model_repository,
            TRITON_CPUS=str(flavor.triton_num_cpu),
            TRITON_MEMORY_DICT=triton_memory,
            TRITON_STORAGE_DICT=triton_storage,
            TRITON_READINESS_INITIAL_DELAY=flavor.triton_readiness_initial_delay_seconds,
            TRITON_COMMAND=triton_command,
            TRITON_COMMIT_TAG=flavor.triton_commit_tag,
        )
    elif endpoint_resource_name == "user-config":
        app_config_serialized = python_json_to_b64(model_bundle.app_config)
        return UserConfigArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            CONFIG_DATA_SERIALIZED=app_config_serialized,
        )
    elif endpoint_resource_name == "endpoint-config":
        endpoint_config_serialized = ModelEndpointConfig(
            endpoint_name=model_endpoint_record.name,
            bundle_name=model_bundle.name,
            post_inference_hooks=build_endpoint_request.post_inference_hooks,
            user_id=user_id,
            billing_queue=hmi_config.billing_queue_arn,
            billing_tags=build_endpoint_request.billing_tags,
            default_callback_url=build_endpoint_request.default_callback_url,
            default_callback_auth=build_endpoint_request.default_callback_auth,
            endpoint_id=model_endpoint_record.id,
            endpoint_type=model_endpoint_record.endpoint_type,
            bundle_id=model_bundle.id,
            labels=build_endpoint_request.labels,
        ).serialize()
        return EndpointConfigArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            ENDPOINT_CONFIG_SERIALIZED=endpoint_config_serialized,
        )
    elif endpoint_resource_name == "horizontal-pod-autoscaler":
        concurrency = get_target_concurrency_from_per_worker_value(
            build_endpoint_request.per_worker
        )
        return HorizontalPodAutoscalerArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            API_VERSION=api_version,
            # Autoscaler arguments
            CONCURRENCY=concurrency,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
        )
    elif endpoint_resource_name == "keda-scaled-object":
        return KedaScaledObjectArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            # Scaled Object arguments
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
            # CONCURRENCY=build_endpoint_request.concurrency,
            REDIS_HOST_PORT=hmi_config.cache_redis_host_port,
            REDIS_DB_INDEX=hmi_config.cache_redis_db_index,
            AUTHENTICATION_REF="azure-workload-identity"
            if infra_config().cloud_provider == "azure"
            else "",
        )
    elif endpoint_resource_name == "service":
        # Use ClusterIP by default for sync endpoint.
        # In Circle CI, we use a NodePort to expose the service to CI.
        service_type = "ClusterIP" if not CIRCLECI else "NodePort"
        if service_type == "NodePort":
            node_port = get_node_port(k8s_resource_group_name)
            node_port_dict = DictStrInt(f"nodePort: {node_port}")
        else:
            node_port_dict = DictStrInt("")
        return ServiceArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            # Service arguments
            NODE_PORT_DICT=node_port_dict,
            SERVICE_TYPE=service_type,
            SERVICE_TARGET_PORT=FORWARDER_PORT,
        )
    elif endpoint_resource_name == "virtual-service":
        return VirtualServiceArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            DNS_HOST_DOMAIN=infra_config().dns_host_domain,
        )
    elif endpoint_resource_name == "destination-rule":
        return DestinationRuleArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
        )
    elif endpoint_resource_name == "vertical-pod-autoscaler":
        return VerticalPodAutoscalerArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
        )
    elif endpoint_resource_name == "pod-disruption-budget":
        return PodDisruptionBudgetArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            GIT_TAG=GIT_TAG,
        )
    else:
        raise Exception(f"Unknown resource name: {endpoint_resource_name}")
