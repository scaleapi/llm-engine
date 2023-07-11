from typing import Dict, Optional, Tuple

from spellbook_serve.common.dtos.model_endpoints import BrokerType
from spellbook_serve.common.dtos.resource_manager import CreateOrUpdateResourcesRequest
from spellbook_serve.core.loggers import filename_wo_ext, make_logger
from spellbook_serve.domain.entities import (
    ModelEndpointInfraState,
    ModelEndpointRecord,
    ModelEndpointType,
)
from spellbook_serve.domain.exceptions import EndpointResourceInvalidRequestException
from spellbook_serve.infra.gateways.resources.endpoint_resource_gateway import (
    EndpointResourceGateway,
    EndpointResourceGatewayCreateOrUpdateResourcesResponse,
    QueueInfo,
)
from spellbook_serve.infra.gateways.resources.k8s_endpoint_resource_delegate import (
    K8SEndpointResourceDelegate,
)
from spellbook_serve.infra.gateways.resources.sqs_endpoint_resource_delegate import (
    SQSEndpointResourceDelegate,
)

logger = make_logger(filename_wo_ext(__file__))


class SqsQueueInfo(QueueInfo):
    """Live endpoints create and use SQS queues. These come with an additional per-queue URL.

    NOTE: broker for this class **MUST** always be SQS.
    """

    queue_url: str

    @staticmethod
    def new(queue_name: str, queue_url: str) -> "SqsQueueInfo":
        return SqsQueueInfo(queue_name=queue_name, broker=BrokerType.SQS, queue_url=queue_url)


class LiveEndpointResourceGateway(EndpointResourceGateway[SqsQueueInfo]):
    def __init__(self, sqs_delegate: SQSEndpointResourceDelegate):
        self.k8s_delegate = K8SEndpointResourceDelegate()
        self.sqs_delegate = sqs_delegate

    async def create_queue(
        self,
        endpoint_record: ModelEndpointRecord,
        labels: Dict[str, str],
    ) -> SqsQueueInfo:
        """Creates a new SQS queue, returning its unique name and queue URL."""
        queue_name, queue_url = await self.sqs_delegate.create_queue_if_not_exists(
            endpoint_id=endpoint_record.id,
            endpoint_name=endpoint_record.name,
            endpoint_created_by=endpoint_record.created_by,
            endpoint_labels=labels,
        )
        return SqsQueueInfo.new(queue_name, queue_url)

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
            destination = f"launch-endpoint-id-{endpoint_record.id.replace('_', '-')}"
            queue_name = None
            queue_url = None

        await self.k8s_delegate.create_or_update_resources(
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
            sqs_attributes = await self.sqs_delegate.get_queue_attributes(endpoint_id=endpoint_id)
            if "ApproximateNumberOfMessages" in sqs_attributes["Attributes"]:
                resources.num_queued_items = int(
                    sqs_attributes["Attributes"]["ApproximateNumberOfMessages"]
                )

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
            await self.sqs_delegate.delete_queue(endpoint_id=endpoint_id)
        except EndpointResourceInvalidRequestException as e:
            logger.warning("Could not delete SQS resources", exc_info=e)
            sqs_result = False

        return k8s_result and sqs_result
