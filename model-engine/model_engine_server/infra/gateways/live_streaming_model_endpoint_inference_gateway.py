from typing import Any, AsyncIterable, Dict, Optional

import aiohttp
import orjson
import requests
import sseclient
from model_engine_server.common.aiohttp_sse_client import EventSource
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
    InvalidRequestException,
    NoHealthyUpstreamException,
    TooManyRequestsException,
    UpstreamServiceError,
)
from model_engine_server.domain.gateways.monitoring_metrics_gateway import MonitoringMetricsGateway
from model_engine_server.domain.gateways.streaming_model_endpoint_inference_gateway import (
    StreamingModelEndpointInferenceGateway,
)
from model_engine_server.infra.gateways.dns_resolver import resolve_dns
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


def _get_streaming_endpoint_url(
    service_name: str, path: str = "/stream", manually_resolve_dns: bool = False
) -> str:
    if CIRCLECI:
        # Circle CI: a NodePort is used to expose the service
        # The IP address is obtained from `minikube ip`.
        protocol: str = "http"
        hostname: str = f"192.168.49.2:{get_node_port(service_name)}"
    elif LOCAL:
        # local development: the svc.cluster.local address is only available w/in the k8s cluster
        protocol = "https"
        hostname = f"{service_name}.{infra_config().dns_host_domain}"
    elif manually_resolve_dns:
        protocol = "http"
        hostname = resolve_dns(
            f"{service_name}.{hmi_config.endpoint_namespace}.svc.cluster.local", port=protocol
        )
    else:
        protocol = "http"
        # no need to hit external DNS resolution if we're w/in the k8s cluster
        hostname = f"{service_name}.{hmi_config.endpoint_namespace}.svc.cluster.local"
    return f"{protocol}://{hostname}{path}"


def _serialize_json(data) -> str:
    # Use orjson, which is faster and more correct than native Python json library.
    # This is more important for sync endpoints, which are more latency-sensitive.
    return orjson.dumps(data).decode()


class LiveStreamingModelEndpointInferenceGateway(StreamingModelEndpointInferenceGateway):
    """
    Concrete implementation for an StreamingModelEndpointInferenceGateway.

    make_single_request() makes the streaming inference request to the endpoint
    make_request_with_retries() wraps make_single_request() with retries
    streaming_predict() wraps make_request_with_retries() and yields SyncEndpointPredictV1Response
    """

    def __init__(self, monitoring_metrics_gateway: MonitoringMetricsGateway, use_asyncio: bool):
        self.monitoring_metrics_gateway = monitoring_metrics_gateway
        self.use_asyncio = use_asyncio

    async def make_single_request(self, request_url: str, payload_json: Dict[str, Any]):
        errored = False
        if self.use_asyncio:
            async with aiohttp.ClientSession(json_serialize=_serialize_json) as aioclient:
                aio_resp = await aioclient.post(
                    request_url,
                    json=payload_json,
                    headers={"Content-Type": "application/json"},
                )
                status = aio_resp.status
                if status == 200:
                    async with EventSource(response=aio_resp) as event_source:
                        async for event in event_source:
                            yield event.data
                else:
                    content = await aio_resp.read()
                    errored = True
        else:
            resp = requests.post(
                request_url,
                json=payload_json,
                headers={"Content-Type": "application/json"},
                stream=True,
            )
            client = sseclient.SSEClient(resp)
            status = resp.status_code
            if status == 200:
                for event in client.events():
                    yield event.data
            else:
                content = resp.content
                errored = True

        # Need to have these exceptions raised outside the async context so that
        # tenacity can properly capture them.
        if errored:
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
        endpoint_name: str,
    ) -> AsyncIterable[Dict[str, Any]]:
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
                    if attempt.retry_state.attempt_number > 1:
                        logger.info(
                            f"Retry number {attempt.retry_state.attempt_number}"
                        )  # pragma: no cover
                    response = self.make_single_request(request_url, payload_json)
                    async for item in response:
                        yield orjson.loads(item)
                    return
        except RetryError as e:
            if isinstance(e.last_attempt.exception(), TooManyRequestsException):
                logger.warning("Hit max # of retries, returning 429 to client")
                self.monitoring_metrics_gateway.emit_http_call_error_metrics(endpoint_name, 429)
                raise UpstreamServiceError(status_code=429, content=b"Too many concurrent requests")
            elif isinstance(e.last_attempt.exception(), NoHealthyUpstreamException):
                logger.warning("Pods didn't spin up in time, returning 503 to client")
                self.monitoring_metrics_gateway.emit_http_call_error_metrics(endpoint_name, 503)
                raise UpstreamServiceError(status_code=503, content=b"No healthy upstream")
            elif isinstance(e.last_attempt.exception(), aiohttp.ClientConnectorError):
                logger.warning("ClientConnectorError, returning 503 to client")
                self.monitoring_metrics_gateway.emit_http_call_error_metrics(endpoint_name, 503)
                raise UpstreamServiceError(status_code=503, content=b"No healthy upstream")
            else:
                logger.error("Unknown Exception Type")
                self.monitoring_metrics_gateway.emit_http_call_error_metrics(endpoint_name, 500)
                raise UpstreamServiceError(status_code=500, content=b"Unknown error")
        except JSONDecodeError:
            logger.exception("JSONDecodeError")
            raise UpstreamServiceError(status_code=500, content=b"JSONDecodeError")

        # Never reached because tenacity should throw a RetryError if we exit the for loop.
        # This is for mypy.
        # pragma: no cover
        raise Exception("Should never reach this line")

    async def streaming_predict(
        self,
        topic: str,
        predict_request: SyncEndpointPredictV1Request,
        manually_resolve_dns: bool = False,
        endpoint_name: Optional[str] = None,
    ) -> AsyncIterable[SyncEndpointPredictV1Response]:
        deployment_url = _get_streaming_endpoint_url(
            topic,
            path=predict_request.destination_path or "/stream",
            manually_resolve_dns=manually_resolve_dns,
        )

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

            response = self.make_request_with_retries(
                request_url=deployment_url,
                payload_json=predict_request.model_dump(exclude_none=True),
                timeout_seconds=timeout_seconds,
                num_retries=num_retries,
                endpoint_name=endpoint_name or topic,
            )
            async for item in response:
                yield SyncEndpointPredictV1Response(
                    status=TaskStatus.SUCCESS, result=item, status_code=200
                )
        except UpstreamServiceError as exc:
            logger.error(f"Service error on streaming task: {exc.content!r}")

            if exc.status_code == 400:
                error_json = orjson.loads(exc.content.decode("utf-8"))
                if "result" in error_json:
                    error_json = orjson.loads(error_json["result"])
                raise InvalidRequestException(error_json)

            try:
                error_json = orjson.loads(exc.content.decode("utf-8"))
                result_traceback = (
                    error_json.get("detail", {}).get("traceback")
                    if isinstance(error_json, dict)
                    else None
                )
            except JSONDecodeError:
                result_traceback = exc.content.decode()

            yield SyncEndpointPredictV1Response(
                status=TaskStatus.FAILURE,
                traceback=result_traceback,
                status_code=exc.status_code,
            )
