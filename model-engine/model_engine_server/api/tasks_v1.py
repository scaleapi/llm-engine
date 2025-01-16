import asyncio

from fastapi import APIRouter, Depends, HTTPException
from model_engine_server.api.dependencies import (
    ExternalInterfaces,
    get_external_interfaces_read_only,
    verify_authentication,
)
from model_engine_server.common.dtos.tasks import (
    CreateAsyncTaskV1Response,
    EndpointPredictV1Request,
    GetAsyncTaskV1Response,
    SyncEndpointPredictV1Request,
    SyncEndpointPredictV1Response,
    TaskStatus,
)
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import (
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


@inference_task_router_v1.post("/async-tasks", response_model=CreateAsyncTaskV1Response)
async def create_async_inference_task(
    model_endpoint_id: str,
    request: EndpointPredictV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
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


@inference_task_router_v1.get("/async-tasks/{task_id}", response_model=GetAsyncTaskV1Response)
def get_async_inference_task(
    task_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> GetAsyncTaskV1Response:
    """
    Gets the status of an async inference task.
    """
    logger.info(f"GET /async-tasks/{task_id} for {auth}")
    try:
        use_case = GetAsyncInferenceTaskV1UseCase(
            model_endpoint_service=external_interfaces.model_endpoint_service,
        )
        return use_case.execute(user=auth, task_id=task_id)
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
