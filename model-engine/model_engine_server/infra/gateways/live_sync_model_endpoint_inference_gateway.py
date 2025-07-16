from typing import Any, Dict, Optional

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
from model_engine_server.core.tracing.tracing_gateway import TracingGateway
from model_engine_server.domain.exceptions import (
    InvalidRequestException,
    NoHealthyUpstreamException,
    TooManyRequestsException,
    UpstreamServiceError,
)
from model_engine_server.domain.gateways.monitoring_metrics_gateway import MonitoringMetricsGateway
from model_engine_server.domain.gateways.sync_model_endpoint_inference_gateway import (
    SyncModelEndpointInferenceGateway,
)
from model_engine_server.infra.gateways.dns_resolver import resolve_dns
from model_engine_server.infra.gateways.k8s_resource_parser import get_node_port
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


def _get_sync_endpoint_url(
    service_name: str, destination_path: str = "/predict", manually_resolve_dns: bool = False
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
    return f"{protocol}://{hostname}{destination_path}"


def _serialize_json(data) -> str:
    # Use orjson, which is faster and more correct than native Python json library.
    # This is more important for sync endpoints, which are more latency-sensitive.
    return orjson.dumps(data).decode()


class LiveSyncModelEndpointInferenceGateway(SyncModelEndpointInferenceGateway):
    """
    Concrete implementation for an SyncModelEndpointInferenceGateway.
    """

    def __init__(
        self,
        monitoring_metrics_gateway: MonitoringMetricsGateway,
        tracing_gateway: TracingGateway,
        use_asyncio: bool,
    ):
        self.monitoring_metrics_gateway = monitoring_metrics_gateway
        self.tracing_gateway = tracing_gateway
        self.use_asyncio = use_asyncio

    async def make_single_request(self, request_url: str, payload_json: Dict[str, Any]):
        headers = {"Content-Type": "application/json"}
        headers.update(self.tracing_gateway.encode_trace_headers())
        if self.use_asyncio:
            async with aiohttp.ClientSession(json_serialize=_serialize_json) as client:
                aio_resp = await client.post(
                    request_url,
                    json=payload_json,
                    headers=headers,
                )
                status = aio_resp.status
                if status == 200:
                    return await aio_resp.json()
                content = await aio_resp.read()
        else:
            resp = requests.post(
                request_url,
                json=payload_json,
                headers=headers,
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
        endpoint_name: str,
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
                    if attempt.retry_state.attempt_number > 1:  # pragma: no cover
                        logger.info(f"Retry number {attempt.retry_state.attempt_number}")
                    with self.tracing_gateway.create_span("make_request_with_retries") as span:
                        span.input = dict(request_url=request_url, payload_json=payload_json)
                        response = await self.make_single_request(request_url, payload_json)
                        span.output = response
                        return response
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

        # Never reached because tenacity should throw a RetryError if we exit the for loop.
        # This is for mypy.
        # pragma: no cover
        return {}

    async def predict(
        self,
        topic: str,
        predict_request: SyncEndpointPredictV1Request,
        manually_resolve_dns: bool = False,
        endpoint_name: Optional[str] = None,
    ) -> SyncEndpointPredictV1Response:
        deployment_url = _get_sync_endpoint_url(
            topic,
            destination_path=predict_request.destination_path or "/predict",
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
            response = await self.make_request_with_retries(
                request_url=deployment_url,
                payload_json=predict_request.model_dump(exclude_none=True),
                timeout_seconds=timeout_seconds,
                num_retries=num_retries,
                endpoint_name=endpoint_name or topic,
            )
        except UpstreamServiceError as exc:
            logger.error(f"Service error on sync task: {exc.content!r}")

            if exc.status_code == 400:
                error_json = orjson.loads(exc.content.decode("utf-8"))
                if "result" in error_json:
                    error_json = orjson.loads(error_json["result"])

                raise InvalidRequestException(error_json)

            try:
                # Try to parse traceback from the response, fallback to just return all the content if failed.
                # Three cases considered:
                # detail.traceback
                # result."detail.traceback"
                # result."detail[]"
                error_json = orjson.loads(exc.content.decode("utf-8"))
                if "result" in error_json:
                    error_json = orjson.loads(error_json["result"])

                detail = error_json.get("detail", {})
                if not isinstance(detail, dict):
                    result_traceback = orjson.dumps(error_json)
                else:
                    result_traceback = error_json.get("detail", {}).get(
                        "traceback", "Failed to parse traceback"
                    )
                return SyncEndpointPredictV1Response(
                    status=TaskStatus.FAILURE,
                    traceback=result_traceback,
                    status_code=exc.status_code,
                )

            except Exception as e:
                logger.error(f"Failed to parse error: {e}")
                return SyncEndpointPredictV1Response(
                    status=TaskStatus.FAILURE,
                    traceback=exc.content.decode(),
                    status_code=exc.status_code,
                )
        return SyncEndpointPredictV1Response(
            status=TaskStatus.SUCCESS, result=response, status_code=200
        )
