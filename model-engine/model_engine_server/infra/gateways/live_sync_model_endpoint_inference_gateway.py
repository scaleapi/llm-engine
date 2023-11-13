from typing import Any, Dict

import aiohttp
import orjson
import requests
from model_engine_server.common.config import hmi_config
from model_engine_server.common.dtos.tasks import (
    SyncEndpointPredictV1Request,
    SyncEndpointPredictV1Response,
    TaskStatus,
)
from model_engine_server.common.env_vars import CIRCLECI, LOCAL
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import (
    NoHealthyUpstreamException,
    TooManyRequestsException,
    UpstreamServiceError,
)
from model_engine_server.domain.gateways.sync_model_endpoint_inference_gateway import (
    SyncModelEndpointInferenceGateway,
)
from model_engine_server.infra.gateways.k8s_resource_parser import get_node_port
from orjson import JSONDecodeError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    stop_any,
    wait_exponential,
)

logger = make_logger(logger_name())

SYNC_ENDPOINT_RETRIES = 8  # Must be an integer >= 0
SYNC_ENDPOINT_MAX_TIMEOUT_SECONDS = 10
SYNC_ENDPOINT_MAX_RETRY_WAIT = 5
SYNC_ENDPOINT_EXP_BACKOFF_BASE = (
    1.2  # Must be a float > 1.0, lower number means more retries but less time waiting.
)


def _get_sync_endpoint_url(deployment_name: str) -> str:
    if CIRCLECI:
        # Circle CI: a NodePort is used to expose the service
        # The IP address is obtained from `minikube ip`.
        protocol: str = "http"
        hostname: str = f"192.168.49.2:{get_node_port(deployment_name)}"
    elif LOCAL:
        # local development: the svc.cluster.local address is only available w/in the k8s cluster
        protocol = "https"
        hostname = f"{deployment_name}.{infra_config().dns_host_domain}"
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
        if status == 503:
            raise NoHealthyUpstreamException("503 returned")
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
                stop=stop_any(
                    stop_after_attempt(num_retries + 1),
                    stop_after_delay(timeout_seconds),
                ),
                retry=retry_if_exception_type(
                    (
                        TooManyRequestsException,
                        NoHealthyUpstreamException,
                        aiohttp.ClientConnectorError,
                    )
                ),
                wait=wait_exponential(
                    multiplier=1,
                    min=1,
                    max=SYNC_ENDPOINT_MAX_RETRY_WAIT,
                    exp_base=SYNC_ENDPOINT_EXP_BACKOFF_BASE,
                ),
            ):
                with attempt:
                    logger.info(f"Retry number {attempt.retry_state.attempt_number}")
                    return await self.make_single_request(request_url, payload_json)
        except RetryError as e:
            if type(e.last_attempt.exception()) == TooManyRequestsException:
                logger.warning("Hit max # of retries, returning 429 to client")
                raise UpstreamServiceError(status_code=429, content=b"Too many concurrent requests")
            elif type(e.last_attempt.exception()) == NoHealthyUpstreamException:
                logger.warning("Pods didn't spin up in time, returning 503 to client")
                raise UpstreamServiceError(status_code=503, content=b"No healthy upstream")
            elif type(e.last_attempt.exception()) == aiohttp.ClientConnectorError:
                logger.warning("ClientConnectorError, returning 503 to client")
                raise UpstreamServiceError(status_code=503, content=b"No healthy upstream")
            else:
                logger.error("Unknown Exception Type")
                raise UpstreamServiceError(status_code=500, content=b"Unknown error")

        # Never reached because tenacity should throw a RetryError if we exit the for loop.
        # This is for mypy.
        # pragma: no cover
        return {}

    async def predict(
        self, topic: str, predict_request: SyncEndpointPredictV1Request
    ) -> SyncEndpointPredictV1Response:
        deployment_url = _get_sync_endpoint_url(topic)

        try:
            timeout_seconds = (
                SYNC_ENDPOINT_MAX_TIMEOUT_SECONDS
                if predict_request.timeout_seconds is None
                else predict_request.timeout_seconds
            )
            num_retries = (
                SYNC_ENDPOINT_RETRIES
                if predict_request.num_retries is None
                else predict_request.num_retries
            )
            response = await self.make_request_with_retries(
                request_url=deployment_url,
                payload_json=predict_request.dict(),
                timeout_seconds=timeout_seconds,
                num_retries=num_retries,
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
