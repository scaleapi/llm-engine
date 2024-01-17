import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from kubernetes_asyncio.client.models.v1_job import V1Job
from kubernetes_asyncio.client.models.v1_pod import V1Pod
from kubernetes_asyncio.client.rest import ApiException
from model_engine_server.common.config import hmi_config
from model_engine_server.common.dtos.batch_jobs import CreateDockerImageBatchJobResourceRequests
from model_engine_server.common.serialization_utils import python_json_to_b64
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import (
    LoggerTagKey,
    LoggerTagManager,
    logger_name,
    make_logger,
)
from model_engine_server.domain.entities.batch_job_entity import BatchJobStatus, DockerImageBatchJob
from model_engine_server.domain.exceptions import EndpointResourceInfraException
from model_engine_server.domain.gateways.docker_image_batch_job_gateway import (
    DockerImageBatchJobGateway,
)
from model_engine_server.infra.gateways.resources.k8s_endpoint_resource_delegate import (
    get_kubernetes_batch_client,
    get_kubernetes_core_client,
    load_k8s_yaml,
    maybe_load_kube_config,
)
from model_engine_server.infra.gateways.resources.k8s_resource_types import (
    DictStrStr,
    DockerImageBatchJobCpuArguments,
    DockerImageBatchJobGpuArguments,
)
from xid import XID

DEFAULT_MOUNT_LOCATION = "/restricted_launch/batch_payload.json"
# Must match resources/docker...{cpu,gpu}.yaml's label selector
LAUNCH_JOB_ID_LABEL_SELECTOR = "launch_job_id"
OWNER_LABEL_SELECTOR = "owner"

ENV: str = os.environ.get("DD_ENV")  # type: ignore
GIT_TAG: str = os.environ.get("GIT_TAG")  # type: ignore
SERVICE_CONFIG_PATH: str = os.environ.get("DEPLOY_SERVICE_CONFIG_PATH")  # type: ignore
DOCKER_IMAGE_BATCH_JOB_SPEC_CPU_PATH = (
    Path(__file__).parent.absolute() / "resources/docker_image_batch_job_cpu.yaml"
)

DOCKER_IMAGE_BATCH_JOB_SPEC_GPU_PATH = (
    Path(__file__).parent.absolute() / "resources/docker_image_batch_job_gpu.yaml"
)

BATCH_JOB_MAX_RUNTIME_SECONDS = 86400 * 7  # 7 days
BATCH_JOB_TTL_SECONDS_AFTER_FINISHED = 86400 * 3  # 3 days

logger = make_logger(logger_name())


class K8sEnvDict(TypedDict):
    name: str
    value: str


def _get_job_id():
    return f"ft-{XID().string()}"


def _check_batch_job_id_valid(job_id: str):
    return re.fullmatch("[a-z0-9-_]*", job_id) is not None


def _add_list_values(
    list_values: List[K8sEnvDict], new_values: List[K8sEnvDict]
) -> List[K8sEnvDict]:
    """
    This function takes a list of dictionaries as input, and adds new name/value pairs to it
    that don't conflict with existing values
    """
    list_values_copy: List[K8sEnvDict] = [x for x in list_values]
    existing_keys = set(dict_["name"] for dict_ in list_values)
    for new_dict in new_values:
        if new_dict["name"] in existing_keys:
            continue
        else:
            list_values_copy.append(new_dict)
    return list_values_copy


def _k8s_job_name_from_id(job_id: str):
    # "di" stands for "docker image" btw
    return f"launch-di-batch-job-{job_id}"


def _parse_job_status_from_k8s_obj(job: V1Job, pods: List[V1Pod]) -> BatchJobStatus:
    status = job.status
    # these counts are the number of pods in some given status
    if status.failed is not None and status.failed > 0:
        return BatchJobStatus.FAILURE
    if status.succeeded is not None and status.succeeded > 0:
        return BatchJobStatus.SUCCESS
    if status.ready is not None and status.ready > 0:
        return BatchJobStatus.RUNNING  # empirically this doesn't happen
    if status.active is not None and status.active > 0:
        for pod in pods:
            # In case there are multiple pods for a given job (e.g. if a pod gets shut down)
            # let's interpret the job as running if any of the pods are running
            # I haven't empirically seen this, but guard against it just in case.
            if pod.status.phase == "Running":
                return BatchJobStatus.RUNNING
        return BatchJobStatus.PENDING
    return BatchJobStatus.PENDING


def make_job_id_to_pods_mapping(pods: List[V1Pod]) -> defaultdict:
    """
    Returns a defaultdict mapping job IDs to pods
    """
    job_id_to_pods_mapping = defaultdict(list)
    for pod in pods:
        job_id = pod.metadata.labels.get(LAUNCH_JOB_ID_LABEL_SELECTOR)
        if job_id is not None:
            job_id_to_pods_mapping[job_id].append(pod)
        else:
            logger.warning(f"Pod {pod.metadata.name} has no job ID label")
    return job_id_to_pods_mapping


class LiveDockerImageBatchJobGateway(DockerImageBatchJobGateway):
    def __init__(self):
        pass

    async def create_docker_image_batch_job(
        self,
        *,
        created_by: str,
        owner: str,
        job_config: Optional[Dict[str, Any]],
        env: Optional[Dict[str, str]],
        command: List[str],
        repo: str,
        tag: str,
        resource_requests: CreateDockerImageBatchJobResourceRequests,
        labels: Dict[str, str],
        mount_location: Optional[str],
        annotations: Optional[Dict[str, str]] = None,
        override_job_max_runtime_s: Optional[int] = None,
        num_workers: Optional[int] = 1,
    ) -> str:
        await maybe_load_kube_config()

        job_id, resource_spec = self._generate_job_spec(
            command=command,
            env=env,
            job_config=job_config,
            mount_location=mount_location,
            repo=repo,
            tag=tag,
            resource_requests=resource_requests,
            created_by=created_by,
            owner=owner,
            labels=labels,
            annotations=annotations,
            override_job_max_runtime_s=override_job_max_runtime_s,
            num_workers=num_workers,
        )
        logger.info(resource_spec)

        batch_client = get_kubernetes_batch_client()

        try:
            await batch_client.create_namespaced_job(
                namespace=hmi_config.endpoint_namespace, body=resource_spec
            )
        except ApiException as exc:
            logger.exception(
                f"Exception encountered when creating batch job on {repo}:{tag} for {owner}"
            )
            raise EndpointResourceInfraException from exc

        return job_id

    @staticmethod
    def _generate_job_spec(
        command: List[str],
        env: Optional[Dict[str, str]],
        job_config: Optional[Dict[str, Any]],
        mount_location: Optional[str],
        repo: str,
        tag: str,
        resource_requests: CreateDockerImageBatchJobResourceRequests,
        created_by: str,
        owner: str,
        labels: Dict[str, str],
        annotations: Optional[Dict[str, str]] = None,
        override_job_max_runtime_s: Optional[int] = None,
        num_workers: Optional[int] = 1,
    ) -> Tuple[str, Dict[str, Any]]:
        job_id = _get_job_id()
        job_name = _k8s_job_name_from_id(job_id)  # why do we even have job_name and id
        job_config_b64encoded = python_json_to_b64(job_config)
        job_runtime_limit = override_job_max_runtime_s or BATCH_JOB_MAX_RUNTIME_SECONDS
        storage = resource_requests.storage
        storage_dict = DictStrStr("")
        if storage is not None:
            storage_dict = DictStrStr(f'ephemeral-storage: "{storage}"')

        if mount_location is None:
            mount_location = DEFAULT_MOUNT_LOCATION
        mount_path = str(Path(mount_location).parent)

        substitution_kwargs: Union[DockerImageBatchJobGpuArguments, DockerImageBatchJobCpuArguments]
        if resource_requests.gpu_type is not None:
            resource_key = "docker-image-batch-job-gpu.yaml"
            substitution_kwargs = DockerImageBatchJobGpuArguments(
                # Base Resource Arguments
                RESOURCE_NAME=job_name,
                NAMESPACE=hmi_config.endpoint_namespace,
                TEAM=labels["team"],
                PRODUCT=labels["product"],
                CREATED_BY=created_by,
                OWNER=owner,
                JOB_ID=job_id,
                GIT_TAG=GIT_TAG,
                # Batch Job Arguments
                BATCH_JOB_MAX_RUNTIME=job_runtime_limit,
                BATCH_JOB_TTL_SECONDS_AFTER_FINISHED=BATCH_JOB_TTL_SECONDS_AFTER_FINISHED,
                IMAGE=f"{infra_config().docker_repo_prefix}/{repo}:{tag}",
                COMMAND=command,
                CPUS=str(resource_requests.cpus),
                MEMORY=str(resource_requests.memory),
                STORAGE_DICT=storage_dict,
                MOUNT_PATH=mount_path,
                INPUT_LOCATION="--input-local",
                # TODO when we enable mounting remote s3files should be "--input-remote"
                S3_FILE="unused",
                LOCAL_FILE_NAME=mount_location,
                FILE_CONTENTS_B64ENCODED=job_config_b64encoded,
                AWS_ROLE=infra_config().profile_ml_inference_worker,
                # GPU Arguments
                GPU_TYPE=resource_requests.gpu_type.value,
                GPUS=resource_requests.gpus or 1,
                REQUEST_ID=LoggerTagManager.get(LoggerTagKey.REQUEST_ID) or "",
                BATCH_JOB_NUM_WORKERS=num_workers or 1,
            )
        else:
            resource_key = "docker-image-batch-job-cpu.yaml"
            substitution_kwargs = DockerImageBatchJobCpuArguments(
                # Base Resource Arguments
                RESOURCE_NAME=job_name,
                NAMESPACE=hmi_config.endpoint_namespace,
                TEAM=labels["team"],
                PRODUCT=labels["product"],
                CREATED_BY=created_by,
                OWNER=owner,
                JOB_ID=job_id,
                GIT_TAG=GIT_TAG,
                # Batch Job Arguments
                BATCH_JOB_MAX_RUNTIME=job_runtime_limit,
                BATCH_JOB_TTL_SECONDS_AFTER_FINISHED=BATCH_JOB_TTL_SECONDS_AFTER_FINISHED,
                IMAGE=f"{infra_config().docker_repo_prefix}/{repo}:{tag}",
                COMMAND=command,
                CPUS=str(resource_requests.cpus),
                MEMORY=str(resource_requests.memory),
                STORAGE_DICT=storage_dict,
                MOUNT_PATH=mount_path,
                INPUT_LOCATION="--input-local",
                # TODO when we enable mounting remote s3files should be "--input-remote"
                S3_FILE="unused",
                LOCAL_FILE_NAME=mount_location,
                FILE_CONTENTS_B64ENCODED=job_config_b64encoded,
                AWS_ROLE=infra_config().profile_ml_inference_worker,
                REQUEST_ID=LoggerTagManager.get(LoggerTagKey.REQUEST_ID) or "",
                BATCH_JOB_NUM_WORKERS=num_workers or 1,
            )

        resource_spec = load_k8s_yaml(resource_key, substitution_kwargs)

        # Only one container for job thankfully
        assert (
            len(resource_spec["spec"]["template"]["spec"]["containers"]) == 1
        ), "Multiple containers found for docker image batch job. Failing build"
        container_env_list = resource_spec["spec"]["template"]["spec"]["containers"][0]["env"]
        if env is None:
            env = {}
        override_envs = [K8sEnvDict(name=name, value=value) for name, value in env.items()]
        resource_spec["spec"]["template"]["spec"]["containers"][0]["env"] = _add_list_values(
            container_env_list, override_envs
        )
        if "annotations" in resource_spec["metadata"]:
            resource_spec["metadata"]["annotations"].update(annotations)
        else:
            resource_spec["metadata"]["annotations"] = annotations
        # add trigger_id label if job was spawned by trigger
        if "trigger_id" in labels:
            resource_spec["metadata"]["labels"]["trigger_id"] = labels["trigger_id"]
        return job_id, resource_spec

    async def get_docker_image_batch_job(self, batch_job_id: str) -> Optional[DockerImageBatchJob]:
        if not _check_batch_job_id_valid(batch_job_id):
            logger.info(f"Invalid batch_job_id passed: {batch_job_id}, returning None")
            return None
        # TODO we can do the auth check here actually
        await maybe_load_kube_config()
        batch_client = get_kubernetes_batch_client()
        try:
            jobs = await batch_client.list_namespaced_job(
                namespace=hmi_config.endpoint_namespace,
                label_selector=f"{LAUNCH_JOB_ID_LABEL_SELECTOR}={batch_job_id}",
            )
            if len(jobs.items) == 0:
                logger.info(f"Job id {batch_job_id} not found")
                return None
            if len(jobs.items) > 1:
                logger.warning(f"Multiple jobs found for id {batch_job_id}")
            job = jobs.items[0]
        except ApiException as exc:
            logger.exception("Got an exception when trying to read the Job")
            raise EndpointResourceInfraException from exc

        core_client = get_kubernetes_core_client()
        try:
            pods = await core_client.list_namespaced_pod(
                namespace=hmi_config.endpoint_namespace,
                label_selector=f"{LAUNCH_JOB_ID_LABEL_SELECTOR}={batch_job_id}",
            )
        except ApiException as exc:
            logger.exception("Got an exception when trying to read pods for the Job")
            raise EndpointResourceInfraException from exc
            # This pod list isn't always needed, but it's simpler code-wise to always make the request

        job_labels = job.metadata.labels
        annotations = job.metadata.annotations

        status = _parse_job_status_from_k8s_obj(job, pods.items)

        return DockerImageBatchJob(
            id=batch_job_id,
            created_by=job_labels.get("created_by"),
            owner=job_labels.get("owner"),
            created_at=job.metadata.creation_timestamp,
            completed_at=job.status.completion_time,
            status=status,
            annotations=annotations,
            num_workers=job.spec.completions,
        )

    async def list_docker_image_batch_jobs(self, owner: str) -> List[DockerImageBatchJob]:
        await maybe_load_kube_config()
        batch_client = get_kubernetes_batch_client()
        try:
            jobs = await batch_client.list_namespaced_job(
                namespace=hmi_config.endpoint_namespace,
                label_selector=f"{OWNER_LABEL_SELECTOR}={owner}",
            )
        except ApiException as exc:
            logger.exception("Got an exception when trying to list the Jobs")
            raise EndpointResourceInfraException from exc

        core_client = get_kubernetes_core_client()
        try:
            pods = await core_client.list_namespaced_pod(
                namespace=hmi_config.endpoint_namespace,
                label_selector=f"{OWNER_LABEL_SELECTOR}={owner},job-name",  # get only pods associated with a job
            )
        except ApiException as exc:
            logger.exception("Got an exception when trying to read pods for the Job")
            raise EndpointResourceInfraException from exc

        # Join jobs + pods
        pods_per_job = make_job_id_to_pods_mapping(pods.items)

        return [
            DockerImageBatchJob(
                id=job.metadata.labels.get(LAUNCH_JOB_ID_LABEL_SELECTOR),
                created_by=job.metadata.labels.get("created_by"),
                owner=owner,
                created_at=job.metadata.creation_timestamp,
                completed_at=job.status.completion_time,
                annotations=job.metadata.annotations,
                status=_parse_job_status_from_k8s_obj(
                    job, pods_per_job[job.metadata.labels.get(LAUNCH_JOB_ID_LABEL_SELECTOR)]
                ),
                num_workers=job.spec.completions,
            )
            for job in jobs.items
        ]

    async def update_docker_image_batch_job(self, batch_job_id: str, cancel: bool) -> bool:
        if cancel:
            return await self._delete_docker_image_batch_job(batch_job_id=batch_job_id)
        return False

    async def _delete_docker_image_batch_job(self, batch_job_id: str) -> bool:
        # tl;dr delete resources corresponding to this batch job
        # simple delete since no configmaps
        job = await self.get_docker_image_batch_job(batch_job_id=batch_job_id)
        if job is None:
            logger.info(f"Job with id {batch_job_id} not found, not doing anything")
            return False
        await maybe_load_kube_config()
        batch_client = get_kubernetes_batch_client()
        job_name = _k8s_job_name_from_id(batch_job_id)
        try:
            # propagation_policy="Background" deletes the corresponding pod as well
            # https://github.com/kubernetes-client/python/issues/234
            await batch_client.delete_namespaced_job(
                name=job_name,
                namespace=hmi_config.endpoint_namespace,
                propagation_policy="Background",
            )
        except ApiException as exc:
            if exc.status == 404:
                logger.info(f"Job id {batch_job_id} does not exist, noop")
                return False
            else:
                logger.exception(
                    f"Got an exception when trying to delete the Job with id {batch_job_id}"
                )
                raise EndpointResourceInfraException from exc
        return True
