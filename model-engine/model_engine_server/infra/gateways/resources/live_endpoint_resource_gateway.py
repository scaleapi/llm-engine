from typing import Dict, Optional, Tuple

from model_engine_server.common.dtos.resource_manager import CreateOrUpdateResourcesRequest
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.entities import (
    ModelEndpointInfraState,
    ModelEndpointRecord,
    ModelEndpointType,
)
from model_engine_server.domain.exceptions import EndpointResourceInfraException
from model_engine_server.domain.gateways import InferenceAutoscalingMetricsGateway
from model_engine_server.infra.gateways.resources.endpoint_resource_gateway import (
    EndpointResourceGateway,
    EndpointResourceGatewayCreateOrUpdateResourcesResponse,
)
from model_engine_server.infra.gateways.resources.k8s_endpoint_resource_delegate import (
    K8SEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.resources.queue_endpoint_resource_delegate import (
    QueueEndpointResourceDelegate,
    QueueInfo,
)

logger = make_logger(logger_name())


class LiveEndpointResourceGateway(EndpointResourceGateway[QueueInfo]):
    def __init__(
        self,
        queue_delegate: QueueEndpointResourceDelegate,
        inference_autoscaling_metrics_gateway: Optional[InferenceAutoscalingMetricsGateway],
    ):
        self.k8s_delegate = K8SEndpointResourceDelegate()
        self.queue_delegate = queue_delegate
        self.inference_autoscaling_metrics_gateway = inference_autoscaling_metrics_gateway

    async def create_queue(
        self,
        endpoint_record: ModelEndpointRecord,
        labels: Dict[str, str],
    ) -> QueueInfo:
        """Creates a new queue, returning its unique name and queue URL."""
        queue_name, queue_url = await self.queue_delegate.create_queue_if_not_exists(
            endpoint_id=endpoint_record.id,
            endpoint_name=endpoint_record.name,
            endpoint_created_by=endpoint_record.created_by,
            endpoint_labels=labels,
        )
        return QueueInfo(queue_name, queue_url)

    async def create_or_update_resources(
        self, request: CreateOrUpdateResourcesRequest
    ) -> EndpointResourceGatewayCreateOrUpdateResourcesResponse:
        endpoint_record = request.build_endpoint_request.model_endpoint_record
        if (
            request.build_endpoint_request.model_endpoint_record.endpoint_type
            == ModelEndpointType.ASYNC
        ):
            q = await self.create_queue(endpoint_record, request.build_endpoint_request.labels)
            queue_name: Optional[str] = q.queue_name
            queue_url: Optional[str] = q.queue_url
            destination: str = q.queue_name
        else:
            # TODO actually just handle the lws thing here
            # destination = f"launch-endpoint-id-{endpoint_record.id.replace('_', '-')}"
            queue_name = None
            queue_url = None

            if self.inference_autoscaling_metrics_gateway is not None:
                await self.inference_autoscaling_metrics_gateway.create_or_update_resources(
                    endpoint_record.id
                )

        destination = await self.k8s_delegate.create_or_update_resources(
            request=request,
            sqs_queue_name=queue_name,
            sqs_queue_url=queue_url,
        )
        return EndpointResourceGatewayCreateOrUpdateResourcesResponse(destination=destination)

    async def get_resources(
        self, endpoint_id: str, deployment_name: str, endpoint_type: ModelEndpointType
    ) -> ModelEndpointInfraState:
        resources = await self.k8s_delegate.get_resources(
            endpoint_id=endpoint_id,
            deployment_name=deployment_name,
            endpoint_type=endpoint_type,
        )

        if endpoint_type == ModelEndpointType.ASYNC:
            sqs_attributes = await self.queue_delegate.get_queue_attributes(endpoint_id=endpoint_id)
            if (
                "Attributes" in sqs_attributes
                and "ApproximateNumberOfMessages" in sqs_attributes["Attributes"]
            ):
                resources.num_queued_items = int(
                    sqs_attributes["Attributes"]["ApproximateNumberOfMessages"]
                )
            elif "active_message_count" in sqs_attributes:  # from ASBQueueEndpointResourceDelegate
                resources.num_queued_items = int(sqs_attributes["active_message_count"])

        return resources

    async def get_all_resources(
        self,
    ) -> Dict[str, Tuple[bool, ModelEndpointInfraState]]:
        return await self.k8s_delegate.get_all_resources()

    async def delete_resources(
        self, endpoint_id: str, deployment_name: str, endpoint_type: ModelEndpointType
    ) -> bool:
        k8s_result = await self.k8s_delegate.delete_resources(
            endpoint_id=endpoint_id,
            deployment_name=deployment_name,
            endpoint_type=endpoint_type,
        )
        sqs_result = True
        try:
            await self.queue_delegate.delete_queue(endpoint_id=endpoint_id)
        except EndpointResourceInfraException as e:
            logger.warning("Could not delete SQS resources", exc_info=e)
            sqs_result = False

        if self.inference_autoscaling_metrics_gateway is not None:
            await self.inference_autoscaling_metrics_gateway.delete_resources(endpoint_id)

        return k8s_result and sqs_result
