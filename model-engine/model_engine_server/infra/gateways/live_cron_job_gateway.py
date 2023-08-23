from typing import Any, Dict, List, Optional

from kubernetes_asyncio.client.rest import ApiException
from model_engine_server.common import dict_not_none
from model_engine_server.common.config import hmi_config
from model_engine_server.core.loggers import filename_wo_ext, make_logger
from model_engine_server.domain.entities.batch_job_entity import DockerImageBatchJob
from model_engine_server.domain.exceptions import EndpointResourceInfraException
from model_engine_server.domain.gateways.cron_job_gateway import CronJobGateway
from model_engine_server.infra.gateways.live_docker_image_batch_job_gateway import (
    LAUNCH_JOB_ID_LABEL_SELECTOR,
    _parse_job_status_from_k8s_obj,
)
from model_engine_server.infra.gateways.resources.k8s_endpoint_resource_delegate import (
    get_kubernetes_batch_client,
    load_k8s_yaml,
    maybe_load_kube_config,
)
from model_engine_server.infra.gateways.resources.k8s_resource_types import CronTriggerArguments

BATCH_CURL_JOB_ACTIVE_DEADLINE_SECONDS = 10

logger = make_logger(filename_wo_ext(__file__))


def _k8s_cron_job_name_from_id(trigger_id: str):
    trigger_id_suffix = trigger_id[5:]  # suffix following "trig_" contains xid
    return f"launch-trigger-{trigger_id_suffix}"


class LiveCronJobGateway(CronJobGateway):
    def __init__(self):
        pass

    async def create_cronjob(
        self,
        *,
        request_host: str,
        trigger_id: str,
        created_by: str,
        owner: str,
        cron_schedule: str,
        docker_image_batch_job_bundle_id: str,
        default_job_config: Optional[Dict[str, Any]],
        default_job_metadata: Dict[str, str],
    ) -> None:
        await maybe_load_kube_config()

        batch_client = get_kubernetes_batch_client()

        cron_job_name = _k8s_cron_job_name_from_id(trigger_id)

        cron_trigger_key = "cron-trigger.yaml"
        substitution_kwargs = CronTriggerArguments(
            HOST=request_host,
            NAME=cron_job_name,
            CREATED_BY=created_by,
            OWNER=owner,
            TEAM=default_job_metadata["team"],
            PRODUCT=default_job_metadata["product"],
            TRIGGER_ID=trigger_id,
            CRON_SCHEDULE=cron_schedule,
            DOCKER_IMAGE_BATCH_JOB_BUNDLE_ID=docker_image_batch_job_bundle_id,
            JOB_CONFIG=self._format_dict_template_args(default_job_config or {}),
            JOB_METADATA=self._format_dict_template_args(default_job_metadata),
            BATCH_CURL_JOB_ACTIVE_DEADLINE_SECONDS=BATCH_CURL_JOB_ACTIVE_DEADLINE_SECONDS,
        )
        cron_job_body = load_k8s_yaml(cron_trigger_key, substitution_kwargs)

        try:
            await batch_client.create_namespaced_cron_job(
                namespace=hmi_config.endpoint_namespace, body=cron_job_body
            )
        except ApiException as exc:
            logger.exception(
                f"Exception encountered when creating batch cron job for docker image batch job bundle id '{docker_image_batch_job_bundle_id}' for {owner}"
            )
            raise EndpointResourceInfraException from exc

    async def list_jobs(
        self,
        *,
        owner: str,
        trigger_id: Optional[str],
    ) -> List[DockerImageBatchJob]:
        await maybe_load_kube_config()

        batch_client = get_kubernetes_batch_client()

        try:
            label_selector = f"trigger_id={trigger_id}" if trigger_id else f"owner={owner}"
            jobs = await batch_client.list_namespaced_job(
                namespace=hmi_config.endpoint_namespace,
                label_selector=label_selector,
            )
        except ApiException as exc:
            logger.exception("Got an exception when trying to list the Jobs")
            raise EndpointResourceInfraException from exc

        return [
            DockerImageBatchJob(
                id=job.metadata.labels.get(LAUNCH_JOB_ID_LABEL_SELECTOR),
                created_by=job.metadata.labels.get("created_by"),
                owner=job.metadata.labels.get("owner"),
                created_at=job.metadata.creation_timestamp,
                completed_at=job.status.completion_time,
                status=_parse_job_status_from_k8s_obj(job),
            )
            for job in jobs.items
        ]

    async def update_cronjob(
        self,
        *,
        trigger_id: str,
        cron_schedule: Optional[str],
        suspend: Optional[bool],
    ) -> None:
        await maybe_load_kube_config()

        batch_client = get_kubernetes_batch_client()

        cron_job_name = _k8s_cron_job_name_from_id(trigger_id)
        partial_body = dict(spec=dict_not_none(schedule=cron_schedule, suspend=suspend))

        try:
            await batch_client.patch_namespaced_cron_job(
                name=cron_job_name,
                namespace=hmi_config.endpoint_namespace,
                body=partial_body,
            )
        except ApiException:
            logger.exception(
                f"Exception encountered when patching batch cron job for trigger id '{trigger_id}', requested object likely does not exist"
            )

    async def delete_cronjob(
        self,
        *,
        trigger_id: str,
    ) -> None:
        await maybe_load_kube_config()

        batch_client = get_kubernetes_batch_client()

        cron_job_name = _k8s_cron_job_name_from_id(trigger_id)

        try:
            await batch_client.delete_namespaced_cron_job(
                name=cron_job_name, namespace=hmi_config.endpoint_namespace
            )
        except ApiException:
            logger.exception(
                f"Exception encountered when deleting batch cron job for trigger id '{trigger_id}', requested object likely does not exist"
            )

    @staticmethod
    def _format_dict_template_args(obj: Dict[str, Any]) -> str:
        return f"{obj}".replace("'", '"')
