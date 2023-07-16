from typing import Dict

from kubernetes_asyncio.client.rest import ApiException
from llm_engine_server.common.config import hmi_config
from llm_engine_server.core.loggers import filename_wo_ext, make_logger
from llm_engine_server.domain.entities import BatchJobSerializationFormat
from llm_engine_server.domain.exceptions import EndpointResourceInfraException
from llm_engine_server.infra.gateways import BatchJobOrchestrationGateway
from llm_engine_server.infra.gateways.live_docker_image_batch_job_gateway import (
    BATCH_JOB_TTL_SECONDS_AFTER_FINISHED,
)
from llm_engine_server.infra.gateways.resources.k8s_endpoint_resource_delegate import (
    get_kubernetes_batch_client,
    load_k8s_yaml,
    maybe_load_kube_config,
)
from llm_engine_server.infra.gateways.resources.k8s_resource_types import (
    BatchJobOrchestrationJobArguments,
)

SHUTDOWN_GRACE_PERIOD = 60

logger = make_logger(filename_wo_ext(__file__))


class LiveBatchJobOrchestrationGateway(BatchJobOrchestrationGateway):
    """
    Live implementation of the BatchJobOrchestrationGateway.
    """

    async def create_batch_job_orchestrator(
        self,
        job_id: str,
        resource_group_name: str,
        owner: str,
        input_path: str,
        serialization_format: BatchJobSerializationFormat,
        labels: Dict[str, str],
        timeout_seconds: float,
    ) -> None:
        await maybe_load_kube_config()

        substitution_kwargs = BatchJobOrchestrationJobArguments(
            RESOURCE_NAME=resource_group_name,
            NAMESPACE=hmi_config.endpoint_namespace,
            TEAM=labels["team"],
            PRODUCT=labels["product"],
            JOB_ID=job_id,
            CREATED_BY=owner,
            OWNER=owner,
            INPUT_LOCATION=input_path,
            SERIALIZATION_FORMAT=serialization_format.value,
            BATCH_JOB_TIMEOUT=timeout_seconds,
            BATCH_JOB_MAX_RUNTIME=int(timeout_seconds + SHUTDOWN_GRACE_PERIOD),
            BATCH_JOB_TTL_SECONDS_AFTER_FINISHED=BATCH_JOB_TTL_SECONDS_AFTER_FINISHED,
        )
        resource_key = "batch-job-orchestration-job.yaml"
        deployment_spec = load_k8s_yaml(resource_key, substitution_kwargs)

        batch_client = get_kubernetes_batch_client()
        try:
            await batch_client.create_namespaced_job(
                namespace=hmi_config.endpoint_namespace, body=deployment_spec
            )
        except ApiException as exc:
            if exc.status == 409:
                logger.info(f"Job {resource_group_name} already exists, replacing")
                await batch_client.replace_namespaced_job(
                    name=resource_group_name,
                    namespace=hmi_config.endpoint_namespace,
                    body=deployment_spec,
                )
            else:
                logger.exception("Got an exception when trying to apply the Job")
                raise EndpointResourceInfraException from exc

    async def delete_batch_job_orchestrator(self, resource_group_name: str) -> bool:
        await maybe_load_kube_config()
        batch_client = get_kubernetes_batch_client()
        try:
            await batch_client.delete_namespaced_job(
                name=resource_group_name, namespace=hmi_config.endpoint_namespace
            )
        except ApiException as exc:
            if exc.status == 404:
                logger.warning(f"Trying to delete nonexistent Job {resource_group_name}.")
            else:
                logger.exception(f"Deletion of Job {resource_group_name} failed.")
                raise EndpointResourceInfraException from exc
        return True
