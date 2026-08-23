import asyncio
import functools

import anyio
from datadog import statsd
from fastapi import APIRouter, Depends, HTTPException, Response
from model_engine_server.api.dependencies import (
    ExternalInterfaces,
    get_external_interfaces_read_only,
    verify_authentication,
)
from model_engine_server.api.rate_limits import user_rate_limit
from model_engine_server.common.dtos.tasks import (
    CreateAsyncTaskV1Response,
    EndpointPredictV1Request,
    GetAsyncTaskV1Response,
    SyncEndpointPredictV1Request,
    SyncEndpointPredictV1Response,
    TaskStatus,
)
from model_engine_server.common.env_vars import DD_ENV
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import (
    BrokerUnavailableException,
    EndpointUnsupportedInferenceTypeException,
    InvalidRequestException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
    UpstreamServiceError,
)
from model_engine_server.domain.use_cases.async_inference_use_cases import (
    CreateAsyncInferenceTaskV1UseCase,
    GetAsyncInferenceTaskV1UseCase,
)
from model_engine_server.domain.use_cases.streaming_inference_use_cases import (
    CreateStreamingInferenceTaskV1UseCase,
)
from model_engine_server.domain.use_cases.sync_inference_use_cases import (
    CreateSyncInferenceTaskV1UseCase,
)
from sse_starlette.sse import EventSourceResponse

inference_task_router_v1 = APIRouter(prefix="/v1")
logger = make_logger(logger_name())


# Task-status polls do a blocking result-backend read (S3 on AWS). They must not share
# the default anyio threadpool: poll volume scales with the number of outstanding tasks,
# and when polls fill the shared pool every other sync route queues behind them.
# Created lazily because anyio.CapacityLimiter requires a running event loop.
@functools.cache
def _get_task_limiter() -> anyio.CapacityLimiter:
    return anyio.CapacityLimiter(40)


@inference_task_router_v1.post("/async-tasks", response_model=CreateAsyncTaskV1Response)
async def create_async_inference_task(
    model_endpoint_id: str,
    request: EndpointPredictV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
    _rate_limit: None = Depends(
        user_rate_limit("post_async_tasks", scope_query_param="model_endpoint_id")
    ),
) -> CreateAsyncTaskV1Response:
    """
    Runs an async inference prediction.
    """
    logger.info(f"POST /async-tasks {request} to endpoint {model_endpoint_id} for {auth}")
    try:
        use_case = CreateAsyncInferenceTaskV1UseCase(
            model_endpoint_service=external_interfaces.model_endpoint_service,
        )
        return await use_case.execute(
            user=auth, model_endpoint_id=model_endpoint_id, request=request
        )
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified endpoint could not be found.",
        ) from exc
    except EndpointUnsupportedInferenceTypeException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported inference type: {str(exc)}",
        ) from exc
    except InvalidRequestException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(exc)}",
        ) from exc
    except BrokerUnavailableException as exc:
        raise HTTPException(
            status_code=503,
            detail="Message broker temporarily unavailable. Please retry.",
        ) from exc


def _emit_result_size(task: GetAsyncTaskV1Response, body: bytes, user_id: str) -> None:
    # Inline result bytes set the cost of the poll route, and neither traces nor
    # access logs capture them; this is the per-tenant payload-size signal.
    statsd.distribution(
        "model_engine.async_task.result_bytes",
        len(body),
        tags=[
            f"env:{DD_ENV}",
            f"user_id:{user_id}",
            f"status:{task.status.value}",
        ],
    )


@inference_task_router_v1.get("/async-tasks/{task_id}", response_model=GetAsyncTaskV1Response)
async def get_async_inference_task(
    task_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
    _rate_limit: None = Depends(user_rate_limit("get_async_task")),
) -> Response:
    """
    Gets the status of an async inference task.
    """
    logger.info(f"GET /async-tasks/{task_id} for {auth}")
    try:
        use_case = GetAsyncInferenceTaskV1UseCase(
            model_endpoint_service=external_interfaces.model_endpoint_service,
        )

        def execute_and_serialize() -> Response:
            task = use_case.execute(user=auth, task_id=task_id)
            # SUCCESS results are unbounded tenant payloads (multi-MB); encoding
            # them on the event loop stalls every request sharing this worker, so
            # the response is serialized here, off the loop, alongside the fetch.
            body = task.model_dump_json().encode("utf-8")
            _emit_result_size(task, body, auth.user_id)
            return Response(content=body, media_type="application/json")

        return await anyio.to_thread.run_sync(
            execute_and_serialize,
            limiter=_get_task_limiter(),
        )
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified task could not be found.",
        ) from exc


@inference_task_router_v1.post("/sync-tasks", response_model=SyncEndpointPredictV1Response)
async def create_sync_inference_task(
    model_endpoint_id: str,
    request: SyncEndpointPredictV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> SyncEndpointPredictV1Response:
    """
    Runs a sync inference prediction.
    """
    logger.info(f"POST /sync-tasks with {request} to endpoint {model_endpoint_id} for {auth}")
    try:
        use_case = CreateSyncInferenceTaskV1UseCase(
            model_endpoint_service=external_interfaces.model_endpoint_service,
        )
        return await use_case.execute(
            user=auth, model_endpoint_id=model_endpoint_id, request=request
        )
    except UpstreamServiceError as exc:
        return SyncEndpointPredictV1Response(
            status=TaskStatus.FAILURE, traceback=exc.content.decode(), status_code=exc.status_code
        )
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified endpoint could not be found.",
        ) from exc
    except EndpointUnsupportedInferenceTypeException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported inference type: {str(exc)}",
        ) from exc
    except asyncio.exceptions.TimeoutError as exc:
        raise HTTPException(
            status_code=408,
            detail="Request timed out.",
        ) from exc
    except InvalidRequestException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(exc)}",
        ) from exc
    except BrokerUnavailableException as exc:
        raise HTTPException(
            status_code=503,
            detail="Message broker temporarily unavailable. Please retry.",
        ) from exc


@inference_task_router_v1.post("/streaming-tasks")
async def create_streaming_inference_task(
    model_endpoint_id: str,
    request: SyncEndpointPredictV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> EventSourceResponse:
    """
    Runs a streaming inference prediction.
    """
    logger.info(f"POST /streaming-tasks with {request} to endpoint {model_endpoint_id} for {auth}")
    try:
        use_case = CreateStreamingInferenceTaskV1UseCase(
            model_endpoint_service=external_interfaces.model_endpoint_service,
        )
        response = await use_case.execute(
            user=auth, model_endpoint_id=model_endpoint_id, request=request
        )

        async def event_generator():
            async for message in response:
                yield {"data": message.json()}

        return EventSourceResponse(event_generator())
    except UpstreamServiceError as exc:
        return EventSourceResponse(
            iter(
                (
                    SyncEndpointPredictV1Response(
                        status=TaskStatus.FAILURE,
                        traceback=exc.content.decode(),
                        status_code=exc.status_code,
                    ).json(),
                )
            )
        )
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified endpoint could not be found.",
        ) from exc
    except EndpointUnsupportedInferenceTypeException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported inference type: {str(exc)}",
        ) from exc
    except InvalidRequestException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(exc)}",
        ) from exc
    except BrokerUnavailableException as exc:
        raise HTTPException(
            status_code=503,
            detail="Message broker temporarily unavailable. Please retry.",
        ) from exc
