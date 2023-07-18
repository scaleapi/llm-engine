import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, TypedDict, Union

from llm_engine_server.common.config import hmi_config
from llm_engine_server.common.dtos.model_endpoints import BrokerName, BrokerType
from llm_engine_server.common.dtos.resource_manager import CreateOrUpdateResourcesRequest
from llm_engine_server.common.env_vars import CIRCLECI
from llm_engine_server.common.resource_limits import (
    FORWARDER_CPU_USAGE,
    FORWARDER_MEMORY_USAGE,
    FORWARDER_STORAGE_USAGE,
)
from llm_engine_server.common.serialization_utils import bool_to_str, python_json_to_b64
from llm_engine_server.core.config import ml_infra_config
from llm_engine_server.domain.entities import (
    ArtifactLike,
    ModelEndpointConfig,
    RunnableImageLike,
    StreamingEnhancedRunnableImageFlavor,
    TritonEnhancedRunnableImageFlavor,
    ZipArtifactFlavor,
)
from llm_engine_server.domain.use_cases.model_endpoint_use_cases import (
    CONVERTED_FROM_ARTIFACT_LIKE_KEY,
)
from llm_engine_server.infra.gateways.k8s_resource_parser import (
    get_node_port,
    get_target_concurrency_from_per_worker_value,
)

__all__: Sequence[str] = (
    "BatchJobOrchestrationJobArguments",
    "CommonEndpointParams",
    "DeploymentArtifactAsyncCpuArguments",
    "DeploymentArtifactAsyncGpuArguments",
    "DeploymentArtifactSyncCpuArguments",
    "DeploymentArtifactSyncGpuArguments",
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
    "LLM_ENGINE_DEFAULT_PRIORITY_CLASS",
    "LLM_ENGINE_HIGH_PRIORITY_CLASS",
    "ResourceArguments",
    "ServiceArguments",
    "UserConfigArguments",
    "VerticalAutoscalingEndpointParams",
    "VerticalPodAutoscalerArguments",
    "VirtualServiceArguments",
    "get_endpoint_resource_arguments_from_request",
)

# Constants for LLMEngine priority classes
LLM_ENGINE_HIGH_PRIORITY_CLASS = "llm-engine-high-priority"
LLM_ENGINE_DEFAULT_PRIORITY_CLASS = "llm-engine-default-priority"

KUBERNETES_MAX_LENGTH = 64
FORWARDER_PORT = 5000
USER_CONTAINER_PORT = 5005
ARTIFACT_LIKE_CONTAINER_PORT = FORWARDER_PORT
FORWARDER_IMAGE_TAG = "0b7ff9d0cdfb27033dd5385c7247a3a38f2829bd"


class _BaseResourceArguments(TypedDict):
    """Keyword-arguments for substituting into all resource templates."""

    RESOURCE_NAME: str
    NAMESPACE: str
    TEAM: str
    PRODUCT: str
    CREATED_BY: str
    OWNER: str


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
    DATADOG_TRACE_ENABLED: str
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


class _StreamingDeploymentArguments(TypedDict):
    """Keyword-arguments for substituting into streaming deployment templates."""

    FORWARDER_PORT: int
    STREAMING_PREDICT_ROUTE: str


class _ArtifactDeploymentArguments(_BaseDeploymentArguments):
    """Keyword-arguments for substituting into artifact deployment templates."""

    BUNDLE_URL: str
    BASE_PATH: str
    LOAD_PREDICT_FN_MODULE_PATH: str
    LOAD_MODEL_FN_MODULE_PATH: str
    CHILD_FN_INFO: str
    PREWARM: str


class _RunnableImageDeploymentArguments(_BaseDeploymentArguments):
    """Keyword-arguments for substituting into runnable image deployment templates."""

    MAIN_ENV: List[Dict[str, Any]]
    COMMAND: List[str]
    PREDICT_ROUTE: str
    HEALTHCHECK_ROUTE: str
    READINESS_INITIAL_DELAY: int
    INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH: str
    FORWARDER_IMAGE_TAG: str
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


class DeploymentArtifactAsyncCpuArguments(_ArtifactDeploymentArguments, _AsyncDeploymentArguments):
    """Keyword-arguments for substituting into CPU async deployment templates with artifacts."""


class DeploymentArtifactAsyncGpuArguments(
    _ArtifactDeploymentArguments, _AsyncDeploymentArguments, _GpuArguments
):
    """Keyword-arguments for substituting into GPU async deployment templates with artifacts."""


class DeploymentArtifactSyncCpuArguments(
    _ArtifactDeploymentArguments, _SyncArtifactDeploymentArguments
):
    """Keyword-arguments for substituting into CPU sync deployment templates with artifacts."""


class DeploymentArtifactSyncGpuArguments(
    _ArtifactDeploymentArguments, _SyncArtifactDeploymentArguments, _GpuArguments
):
    """Keyword-arguments for substituting into GPU sync deployment templates with artifacts."""


class DeploymentRunnableImageSyncCpuArguments(
    _RunnableImageDeploymentArguments, _SyncRunnableImageDeploymentArguments
):
    """Keyword-arguments for substituting into CPU sync deployment templates for runnable images."""


class DeploymentRunnableImageSyncGpuArguments(
    _RunnableImageDeploymentArguments,
    _SyncRunnableImageDeploymentArguments,
    _GpuArguments,
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
    _RunnableImageDeploymentArguments,
    _SyncRunnableImageDeploymentArguments,
    _TritonArguments,
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
    _RunnableImageDeploymentArguments,
    _AsyncDeploymentArguments,
    _GpuArguments,
    _TritonArguments,
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
    DeploymentArtifactAsyncCpuArguments,
    DeploymentArtifactAsyncGpuArguments,
    DeploymentArtifactSyncCpuArguments,
    DeploymentArtifactSyncGpuArguments,
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
]


def container_start_triton_cmd(
    triton_model_repository: str,
    triton_model_replicas: Dict[str, int],
    ipv6_healthcheck: bool = False,
) -> List[str]:
    # NOTE: this path is set in the Trtion-specific Dockerfile:
    # std-ml-srv/ml_serve/triton/Dockerfile
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
    api_version: Optional[str] = None,
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
    prewarm = bool_to_str(build_endpoint_request.prewarm) or "false"
    sqs_profile = "default"  # TODO: Make this configurable
    s3_bucket = ml_infra_config().s3_bucket

    load_predict_fn_module_path = ""
    load_model_fn_module_path = ""
    if isinstance(flavor, ZipArtifactFlavor):
        load_predict_fn_module_path = flavor.load_predict_fn_module_path
        load_model_fn_module_path = flavor.load_model_fn_module_path

    storage_dict = DictStrStr("")
    if storage is not None:
        storage_dict = DictStrStr(f'ephemeral-storage: "{storage}"')

    change_cause_message = (
        f"Deployment at {datetime.utcnow()} UTC. "
        f"Using deployment constructed from model bundle ID: {model_bundle.id}, "
        f"model bundle name: {model_bundle.name}, "
        f"endpoint ID: {model_endpoint_record.id}"
    )

    priority = LLM_ENGINE_DEFAULT_PRIORITY_CLASS
    if build_endpoint_request.high_priority:
        priority = LLM_ENGINE_HIGH_PRIORITY_CLASS

    image_hash = str(hashlib.md5(str(request.image).encode()).hexdigest())[:KUBERNETES_MAX_LENGTH]

    # In Circle CI, we use Redis on localhost instead of SQS
    broker_name = BrokerName.SQS.value if not CIRCLECI else BrokerName.REDIS.value
    broker_type = BrokerType.SQS.value if not CIRCLECI else BrokerType.REDIS.value
    datadog_trace_enabled = hmi_config.datadog_trace_enabled
    if broker_type == BrokerType.REDIS.value:
        sqs_queue_url = ""

    main_env = []
    if isinstance(flavor, RunnableImageLike) and flavor.env:
        main_env = [{"name": key, "value": value} for key, value in flavor.env.items()]

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
        infra_service_config_volume_mount_path = f"{flavor.env['BASE_PATH']}/ml_infra_core/llm_engine_server.core/llm_engine_server.core/configs"
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
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DATADOG_TRACE_ENABLED=datadog_trace_enabled,
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
            FORWARDER_IMAGE_TAG=FORWARDER_IMAGE_TAG,
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
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DATADOG_TRACE_ENABLED=datadog_trace_enabled,
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
            FORWARDER_IMAGE_TAG=FORWARDER_IMAGE_TAG,
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
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DATADOG_TRACE_ENABLED=datadog_trace_enabled,
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
            FORWARDER_IMAGE_TAG=FORWARDER_IMAGE_TAG,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Streaming Deployment Arguments
            FORWARDER_PORT=FORWARDER_PORT,
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
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DATADOG_TRACE_ENABLED=datadog_trace_enabled,
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
            FORWARDER_IMAGE_TAG=FORWARDER_IMAGE_TAG,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Streaming Deployment Arguments
            FORWARDER_PORT=FORWARDER_PORT,
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
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DATADOG_TRACE_ENABLED=datadog_trace_enabled,
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
            FORWARDER_IMAGE_TAG=FORWARDER_IMAGE_TAG,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Sync Deployment Arguments
            FORWARDER_PORT=FORWARDER_PORT,
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
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DATADOG_TRACE_ENABLED=datadog_trace_enabled,
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
            FORWARDER_IMAGE_TAG=FORWARDER_IMAGE_TAG,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Sync Deployment Arguments
            FORWARDER_PORT=FORWARDER_PORT,
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
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DATADOG_TRACE_ENABLED=datadog_trace_enabled,
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
            FORWARDER_IMAGE_TAG=FORWARDER_IMAGE_TAG,
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
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DATADOG_TRACE_ENABLED=datadog_trace_enabled,
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
            FORWARDER_IMAGE_TAG=FORWARDER_IMAGE_TAG,
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
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DATADOG_TRACE_ENABLED=datadog_trace_enabled,
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
            FORWARDER_IMAGE_TAG=FORWARDER_IMAGE_TAG,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Sync Deployment Arguments
            FORWARDER_PORT=FORWARDER_PORT,
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
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DATADOG_TRACE_ENABLED=datadog_trace_enabled,
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
            FORWARDER_IMAGE_TAG=FORWARDER_IMAGE_TAG,
            INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH=infra_service_config_volume_mount_path,
            FORWARDER_CONFIG_FILE_NAME=forwarder_config_file_name,
            FORWARDER_CPUS_LIMIT=FORWARDER_CPU_USAGE,
            FORWARDER_MEMORY_LIMIT=FORWARDER_MEMORY_USAGE,
            FORWARDER_STORAGE_LIMIT=FORWARDER_STORAGE_USAGE,
            USER_CONTAINER_PORT=USER_CONTAINER_PORT,
            # Sync Deployment Arguments
            FORWARDER_PORT=FORWARDER_PORT,
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
    elif endpoint_resource_name == "deployment-artifact-async-cpu":
        assert isinstance(flavor, ArtifactLike)
        return DeploymentArtifactAsyncCpuArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DATADOG_TRACE_ENABLED=datadog_trace_enabled,
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
            STORAGE_DICT=storage_dict,
            BASE_PATH="/app",
            PER_WORKER=build_endpoint_request.per_worker,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
            RESULTS_S3_BUCKET=s3_bucket,
            # Artifact Arguments
            BUNDLE_URL=flavor.location,
            LOAD_PREDICT_FN_MODULE_PATH=load_predict_fn_module_path,
            LOAD_MODEL_FN_MODULE_PATH=load_model_fn_module_path,
            CHILD_FN_INFO=json.dumps(
                build_endpoint_request.child_fn_info if build_endpoint_request.child_fn_info else {}
            ),
            PREWARM=prewarm,
            # Async Deployment Arguments
            CELERY_S3_BUCKET=s3_bucket,
            QUEUE=sqs_queue_name,
            BROKER_NAME=broker_name,
            BROKER_TYPE=broker_type,
            SQS_QUEUE_URL=sqs_queue_url,
            SQS_PROFILE=sqs_profile,
        )
    elif endpoint_resource_name == "deployment-artifact-async-gpu":
        assert isinstance(flavor, ArtifactLike)
        assert build_endpoint_request.gpu_type is not None
        return DeploymentArtifactAsyncGpuArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DATADOG_TRACE_ENABLED=datadog_trace_enabled,
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
            STORAGE_DICT=storage_dict,
            BASE_PATH="/app",
            PER_WORKER=build_endpoint_request.per_worker,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
            RESULTS_S3_BUCKET=s3_bucket,
            # Artifact Arguments
            BUNDLE_URL=flavor.location,
            LOAD_PREDICT_FN_MODULE_PATH=load_predict_fn_module_path,
            LOAD_MODEL_FN_MODULE_PATH=load_model_fn_module_path,
            CHILD_FN_INFO=json.dumps(
                build_endpoint_request.child_fn_info if build_endpoint_request.child_fn_info else {}
            ),
            PREWARM=prewarm,
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
    elif endpoint_resource_name == "deployment-artifact-sync-cpu":
        assert isinstance(flavor, ArtifactLike)
        return DeploymentArtifactSyncCpuArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DATADOG_TRACE_ENABLED=datadog_trace_enabled,
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
            STORAGE_DICT=storage_dict,
            BASE_PATH="/app",
            PER_WORKER=build_endpoint_request.per_worker,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
            RESULTS_S3_BUCKET=s3_bucket,
            # Artifact Arguments
            BUNDLE_URL=flavor.location,
            LOAD_PREDICT_FN_MODULE_PATH=load_predict_fn_module_path,
            LOAD_MODEL_FN_MODULE_PATH=load_model_fn_module_path,
            CHILD_FN_INFO=json.dumps(
                build_endpoint_request.child_fn_info if build_endpoint_request.child_fn_info else {}
            ),
            PREWARM=prewarm,
            # Sync Artifact DeploymentArguments Arguments
            ARTIFACT_LIKE_CONTAINER_PORT=ARTIFACT_LIKE_CONTAINER_PORT,
        )
    elif endpoint_resource_name == "deployment-artifact-sync-gpu":
        assert isinstance(flavor, ArtifactLike)
        assert build_endpoint_request.gpu_type is not None
        return DeploymentArtifactSyncGpuArguments(
            # Base resource arguments
            RESOURCE_NAME=k8s_resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            ENDPOINT_ID=model_endpoint_record.id,
            ENDPOINT_NAME=model_endpoint_record.name,
            TEAM=team,
            PRODUCT=product,
            CREATED_BY=created_by,
            OWNER=owner,
            # Base deployment arguments
            CHANGE_CAUSE_MESSAGE=change_cause_message,
            AWS_ROLE=build_endpoint_request.aws_role,
            PRIORITY=priority,
            IMAGE=request.image,
            IMAGE_HASH=image_hash,
            DATADOG_TRACE_ENABLED=datadog_trace_enabled,
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
            STORAGE_DICT=storage_dict,
            BASE_PATH="/app",
            PER_WORKER=build_endpoint_request.per_worker,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
            RESULTS_S3_BUCKET=s3_bucket,
            # Artifact Arguments
            BUNDLE_URL=flavor.location,
            LOAD_PREDICT_FN_MODULE_PATH=load_predict_fn_module_path,
            LOAD_MODEL_FN_MODULE_PATH=load_model_fn_module_path,
            CHILD_FN_INFO=json.dumps(
                build_endpoint_request.child_fn_info if build_endpoint_request.child_fn_info else {}
            ),
            PREWARM=prewarm,
            # Sync Artifact DeploymentArguments Arguments
            ARTIFACT_LIKE_CONTAINER_PORT=ARTIFACT_LIKE_CONTAINER_PORT,
            # GPU Deployment Arguments
            GPU_TYPE=build_endpoint_request.gpu_type.value,
            GPUS=build_endpoint_request.gpus,
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
            CONFIG_DATA_SERIALIZED=app_config_serialized,
        )
    elif endpoint_resource_name == "endpoint-config":
        endpoint_config_serialized = ModelEndpointConfig(
            endpoint_name=model_endpoint_record.name,
            bundle_name=model_bundle.name,
            post_inference_hooks=build_endpoint_request.post_inference_hooks,
            user_id=user_id,
            default_callback_url=build_endpoint_request.default_callback_url,
            default_callback_auth=build_endpoint_request.default_callback_auth,
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
            API_VERSION=api_version,
            # Autoscaler arguments
            CONCURRENCY=concurrency,
            MIN_WORKERS=build_endpoint_request.min_workers,
            MAX_WORKERS=build_endpoint_request.max_workers,
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
            DNS_HOST_DOMAIN=ml_infra_config().dns_host_domain,
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
            CPUS=str(build_endpoint_request.cpus),
            MEMORY=str(build_endpoint_request.memory),
        )
    else:
        raise Exception(f"Unknown resource name: {endpoint_resource_name}")
