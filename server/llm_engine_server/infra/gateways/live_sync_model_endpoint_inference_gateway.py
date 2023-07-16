from typing import Any, Dict

import aiohttp
import orjson
import requests
from orjson import JSONDecodeError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_engine_server.common.config import hmi_config
from llm_engine_server.common.dtos.tasks import (
    EndpointPredictV1Request,
    SyncEndpointPredictV1Response,
    TaskStatus,
)
from llm_engine_server.common.env_vars import CIRCLECI, LOCAL
from llm_engine_server.core.config import ml_infra_config
from llm_engine_server.core.loggers import filename_wo_ext, make_logger
from llm_engine_server.domain.exceptions import TooManyRequestsException, UpstreamServiceError
from llm_engine_server.domain.gateways.sync_model_endpoint_inference_gateway import (
    SyncModelEndpointInferenceGateway,
)
from llm_engine_server.infra.gateways.k8s_resource_parser import get_node_port

logger = make_logger(filename_wo_ext(__file__))

SYNC_ENDPOINT_RETRIES = 5  # Must be an integer >= 0
SYNC_ENDPOINT_MAX_TIMEOUT_SECONDS = 10


def _get_sync_endpoint_url(deployment_name: str) -> str:
    if CIRCLECI:
        # Circle CI: a NodePort is used to expose the service
        # The IP address is obtained from `minikube ip`.
        protocol: str = "http"
        hostname: str = f"192.168.49.2:{get_node_port(deployment_name)}"
    elif LOCAL:
        # local development: the svc.cluster.local address is only available w/in the k8s cluster
        protocol = "https"
        hostname = f"{deployment_name}.{ml_infra_config().dns_host_domain}"
    else:
        protocol = "http"
        # no need to hit external DNS resolution if we're w/in the k8s cluster
        hostname = f"{deployment_name}.{hmi_config.endpoint_namespace}.svc.cluster.local"
    return f"{protocol}://{hostname}/predict"


def _serialize_json(data) -> str:
    # Use orjson, which is faster and more correct than native Python json library.
    # This is more important for sync endpoints, which are more latency-sensitive.
    return orjson.dumps(data).decode()


class LiveSyncModelEndpointInferenceGateway(SyncModelEndpointInferenceGateway):
    """
    Concrete implementation for an SyncModelEndpointInferenceGateway.
    """

    def __init__(self, use_asyncio: bool):
        self.use_asyncio = use_asyncio

    async def make_single_request(self, request_url: str, payload_json: Dict[str, Any]):
        if self.use_asyncio:
            async with aiohttp.ClientSession(json_serialize=_serialize_json) as client:
                aio_resp = await client.post(
                    request_url,
                    json=payload_json,
                    headers={"Content-Type": "application/json"},
                )
                status = aio_resp.status
                if status == 200:
                    return await aio_resp.json()
                content = await aio_resp.read()
        else:
            resp = requests.post(
                request_url,
                json=payload_json,
                headers={"Content-Type": "application/json"},
            )
            status = resp.status_code
            if status == 200:
                return resp.json()
            content = resp.content

        # Need to have these exceptions raised outside the async context so that
        # tenacity can properly capture them.
        if status == 429:
            raise TooManyRequestsException("429 returned")
        else:
            raise UpstreamServiceError(status_code=status, content=content)

    async def make_request_with_retries(
        self,
        request_url: str,
        payload_json: Dict[str, Any],
        timeout_seconds: float,
        num_retries: int,
    ) -> Dict[str, Any]:
        # Copied from document-endpoint
        # More details at https://tenacity.readthedocs.io/en/latest/#retrying-code-block
        # Try/catch + for loop makes us retry only when we get a 429 from the synchronous endpoint.
        # We should be creating a new requests Session each time, which should avoid sending
        # requests to the same endpoint. This is admittedly a hack until we get proper
        # least-outstanding-requests load balancing to our http endpoints

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(num_retries + 1),
                retry=retry_if_exception_type(TooManyRequestsException),
                wait=wait_exponential(multiplier=1, min=1, max=timeout_seconds),
            ):
                with attempt:
                    logger.info(f"Retry number {attempt.retry_state.attempt_number}")
                    return await self.make_single_request(request_url, payload_json)
        except RetryError:
            logger.warning("Hit max # of retries, returning 429 to client")
            raise UpstreamServiceError(status_code=429, content=b"Too many concurrent requests")

        # Never reached because tenacity should throw a RetryError if we exit the for loop.
        # This is for mypy.
        # pragma: no cover
        return {}

    async def predict(
        self, topic: str, predict_request: EndpointPredictV1Request
    ) -> SyncEndpointPredictV1Response:
        deployment_url = _get_sync_endpoint_url(topic)

        try:
            response = await self.make_request_with_retries(
                request_url=deployment_url,
                payload_json=predict_request.dict(),
                timeout_seconds=SYNC_ENDPOINT_MAX_TIMEOUT_SECONDS,
                num_retries=SYNC_ENDPOINT_RETRIES,
            )
        except UpstreamServiceError as exc:
            logger.error(f"Service error on sync task: {exc.content!r}")
            try:
                error_json = orjson.loads(exc.content.decode("utf-8"))
                result_traceback = (
                    error_json.get("detail", {}).get("traceback")
                    if isinstance(error_json, dict)
                    else None
                )
                return SyncEndpointPredictV1Response(
                    status=TaskStatus.FAILURE,
                    traceback=result_traceback,
                )
            except JSONDecodeError:
                return SyncEndpointPredictV1Response(
                    status=TaskStatus.FAILURE, traceback=exc.content.decode()
                )

        return SyncEndpointPredictV1Response(status=TaskStatus.SUCCESS, result=response)
