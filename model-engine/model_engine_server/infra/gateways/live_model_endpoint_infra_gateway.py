import os
from typing import Any, Dict, List, Optional

from model_engine_server.common.dtos.endpoint_builder import BuildEndpointRequest
from model_engine_server.common.settings import (
    RESTRICTED_ENDPOINT_LABELS,
    generate_deployment_name,
    get_service_builder_queue,
)
from model_engine_server.domain.entities import (
    CallbackAuth,
    CpuSpecificationType,
    GpuType,
    ModelEndpointInfraState,
    ModelEndpointRecord,
    StorageSpecificationType,
)
from model_engine_server.domain.exceptions import EndpointResourceInfraException
from model_engine_server.domain.gateways import TaskQueueGateway
from model_engine_server.infra.gateways.model_endpoint_infra_gateway import (
    ModelEndpointInfraGateway,
)
from model_engine_server.infra.gateways.resources.endpoint_resource_gateway import (
    EndpointResourceGateway,
)

BUILD_TASK_NAME = "model_engine_server.service_builder.tasks_v1.build_endpoint"
SERVICE_IDENTIFIER = os.getenv("SERVICE_IDENTIFIER")


def redact_restricted_labels(labels: Dict[str, str]) -> None:
    """
    When editing an endpoint, we'll simply read all of the existing labels from the endpoint first
    to establish the initial state. The wrinkle there is that some of these existing labels are
    themselves "restricted", in the sense that they are derived data that cannot be passed in
    again from the API. Therefore, let's scrub those out.
    """

    for key in RESTRICTED_ENDPOINT_LABELS:
        if key in labels:
            del labels[key]


class LiveModelEndpointInfraGateway(ModelEndpointInfraGateway):
    def __init__(
        self,
        resource_gateway: EndpointResourceGateway,
        task_queue_gateway: TaskQueueGateway,
    ):
        self.resource_gateway = resource_gateway
        self.task_queue_gateway = task_queue_gateway

    def create_model_endpoint_infra(
        self,
        *,
        model_endpoint_record: ModelEndpointRecord,
        min_workers: int,
        max_workers: int,
        per_worker: int,
        cpus: CpuSpecificationType,
        gpus: int,
        memory: StorageSpecificationType,
        gpu_type: Optional[GpuType],
        storage: Optional[StorageSpecificationType],
        optimize_costs: bool,
        aws_role: str,
        results_s3_bucket: str,
        child_fn_info: Optional[Dict[str, Any]],
        post_inference_hooks: Optional[List[str]],
        labels: Dict[str, str],
        prewarm: bool,
        high_priority: Optional[bool],
        billing_tags: Optional[Dict[str, Any]] = None,
        default_callback_url: Optional[str],
        default_callback_auth: Optional[CallbackAuth],
        queue_message_timeout_duration: Optional[int] = None,
    ) -> str:
        deployment_name = generate_deployment_name(
            model_endpoint_record.created_by, model_endpoint_record.name
        )
        build_endpoint_request = BuildEndpointRequest(
            model_endpoint_record=model_endpoint_record,
            deployment_name=deployment_name,
            min_workers=min_workers,
            max_workers=max_workers,
            per_worker=per_worker,
            cpus=cpus,
            gpus=gpus,
            memory=memory,
            gpu_type=gpu_type,
            storage=storage,
            optimize_costs=optimize_costs,
            aws_role=aws_role,
            results_s3_bucket=results_s3_bucket,
            child_fn_info=child_fn_info,
            post_inference_hooks=post_inference_hooks,
            labels=labels,
            prewarm=prewarm,
            high_priority=high_priority,
            billing_tags=billing_tags,
            default_callback_url=default_callback_url,
            default_callback_auth=default_callback_auth,
            queue_message_timeout_duration=queue_message_timeout_duration,
        )
        response = self.task_queue_gateway.send_task(
            task_name=BUILD_TASK_NAME,
            queue_name=get_service_builder_queue(SERVICE_IDENTIFIER),
            # celery request is required to be JSON serializables
            kwargs=dict(build_endpoint_request_json=build_endpoint_request.dict()),
        )
        return response.task_id

    async def update_model_endpoint_infra(
        self,
        *,
        model_endpoint_record: ModelEndpointRecord,
        min_workers: Optional[int] = None,
        max_workers: Optional[int] = None,
        per_worker: Optional[int] = None,
        cpus: Optional[CpuSpecificationType] = None,
        gpus: Optional[int] = None,
        memory: Optional[StorageSpecificationType] = None,
        gpu_type: Optional[GpuType] = None,
        storage: Optional[StorageSpecificationType] = None,
        optimize_costs: Optional[bool] = None,
        child_fn_info: Optional[Dict[str, Any]] = None,
        post_inference_hooks: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        prewarm: Optional[bool] = None,
        high_priority: Optional[bool] = None,
        billing_tags: Optional[Dict[str, Any]] = None,
        default_callback_url: Optional[str] = None,
        default_callback_auth: Optional[CallbackAuth] = None,
    ) -> str:
        infra_state = await self.get_model_endpoint_infra(
            model_endpoint_record=model_endpoint_record
        )
        if infra_state is None:
            raise EndpointResourceInfraException
        if min_workers is None:
            min_workers = infra_state.deployment_state.min_workers
        if max_workers is None:
            max_workers = infra_state.deployment_state.max_workers
        if per_worker is None:
            per_worker = infra_state.deployment_state.per_worker
        if cpus is None:
            cpus = infra_state.resource_state.cpus
        if gpus is None:
            gpus = infra_state.resource_state.gpus
        if memory is None:
            memory = infra_state.resource_state.memory
        if gpu_type is None:
            gpu_type = infra_state.resource_state.gpu_type
        if storage is None:
            storage = infra_state.resource_state.storage
        if optimize_costs is None:
            optimize_costs = infra_state.resource_state.optimize_costs or False
        if child_fn_info is None:
            child_fn_info = infra_state.child_fn_info
        endpoint_config = infra_state.user_config_state.endpoint_config
        if post_inference_hooks is None and endpoint_config is not None:
            post_inference_hooks = endpoint_config.post_inference_hooks
        if labels is None:
            labels = infra_state.labels
        else:
            infra_state.labels.update(labels)
            labels = infra_state.labels
        assert labels is not None
        if billing_tags is None and endpoint_config is not None:
            billing_tags = endpoint_config.billing_tags
        redact_restricted_labels(labels)
        if prewarm is None:
            if infra_state.prewarm is None:
                # just update old endpoints to use prewarming
                prewarm = True
            else:
                prewarm = infra_state.prewarm
        if high_priority is None:
            # Can only happen for old endpoints
            if infra_state.high_priority is None:
                # Update old endpoints to default priority
                high_priority = False
            else:
                high_priority = infra_state.high_priority
        if default_callback_url is None and endpoint_config is not None:
            default_callback_url = endpoint_config.default_callback_url
        if default_callback_auth is None and endpoint_config is not None:
            default_callback_auth = endpoint_config.default_callback_auth

        aws_role = infra_state.aws_role
        results_s3_bucket = infra_state.results_s3_bucket

        build_endpoint_request = BuildEndpointRequest(
            model_endpoint_record=model_endpoint_record,
            deployment_name=infra_state.deployment_name,
            min_workers=min_workers,
            max_workers=max_workers,
            per_worker=per_worker,
            cpus=cpus,
            gpus=gpus,
            memory=memory,
            gpu_type=gpu_type,
            storage=storage,
            optimize_costs=optimize_costs,
            aws_role=aws_role,
            results_s3_bucket=results_s3_bucket,
            child_fn_info=child_fn_info,
            post_inference_hooks=post_inference_hooks,
            labels=labels,
            prewarm=prewarm,
            high_priority=high_priority,
            billing_tags=billing_tags,
            default_callback_url=default_callback_url,
            default_callback_auth=default_callback_auth,
        )
        response = self.task_queue_gateway.send_task(
            task_name=BUILD_TASK_NAME,
            queue_name=get_service_builder_queue(SERVICE_IDENTIFIER),
            kwargs=dict(build_endpoint_request_json=build_endpoint_request.dict()),
        )
        return response.task_id

    async def get_model_endpoint_infra(
        self, model_endpoint_record: ModelEndpointRecord
    ) -> Optional[ModelEndpointInfraState]:
        deployment_name = generate_deployment_name(
            model_endpoint_record.created_by, model_endpoint_record.name
        )
        try:
            return await self.resource_gateway.get_resources(
                endpoint_id=model_endpoint_record.id,
                deployment_name=deployment_name,
                endpoint_type=model_endpoint_record.endpoint_type,
            )
        except EndpointResourceInfraException:
            return None

    async def delete_model_endpoint_infra(self, model_endpoint_record: ModelEndpointRecord) -> bool:
        deployment_name = generate_deployment_name(
            model_endpoint_record.created_by, model_endpoint_record.name
        )
        endpoint_type = model_endpoint_record.endpoint_type
        return await self.resource_gateway.delete_resources(
            endpoint_id=model_endpoint_record.id,
            deployment_name=deployment_name,
            endpoint_type=endpoint_type,
        )
