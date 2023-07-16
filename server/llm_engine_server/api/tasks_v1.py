from fastapi import APIRouter, Depends, HTTPException
from llm_engine_server.api.dependencies import (
    ExternalInterfaces,
    get_external_interfaces_read_only,
    verify_authentication,
)
from llm_engine_server.common.datadog_utils import add_trace_resource_name
from llm_engine_server.common.dtos.tasks import (
    CreateAsyncTaskV1Response,
    EndpointPredictV1Request,
    GetAsyncTaskV1Response,
    SyncEndpointPredictV1Response,
    TaskStatus,
)
from llm_engine_server.core.auth.authentication_repository import User
from llm_engine_server.core.domain_exceptions import (
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from llm_engine_server.core.loggers import filename_wo_ext, make_logger
from llm_engine_server.domain.exceptions import (
    EndpointUnsupportedInferenceTypeException,
    UpstreamServiceError,
)
from llm_engine_server.domain.use_cases.async_inference_use_cases import (
    CreateAsyncInferenceTaskV1UseCase,
    GetAsyncInferenceTaskV1UseCase,
)
from llm_engine_server.domain.use_cases.streaming_inference_use_cases import (
    CreateStreamingInferenceTaskV1UseCase,
)
from llm_engine_server.domain.use_cases.sync_inference_use_cases import (
    CreateSyncInferenceTaskV1UseCase,
)
from sse_starlette.sse import EventSourceResponse

inference_task_router_v1 = APIRouter(prefix="/v1")
logger = make_logger(filename_wo_ext(__name__))


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
    add_trace_resource_name("task_async_post")
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


@inference_task_router_v1.get("/async-tasks/{task_id}", response_model=GetAsyncTaskV1Response)
def get_async_inference_task(
    task_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> GetAsyncTaskV1Response:
    """
    Gets the status of an async inference task.
    """
    add_trace_resource_name("task_async_id_get")
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
    request: EndpointPredictV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> SyncEndpointPredictV1Response:
    """
    Runs a sync inference prediction.
    """
    add_trace_resource_name("task_sync_post")
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
            status=TaskStatus.FAILURE, traceback=exc.content.decode()
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


@inference_task_router_v1.post("/streaming-tasks")
async def create_streaming_inference_task(
    model_endpoint_id: str,
    request: EndpointPredictV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> EventSourceResponse:
    """
    Runs a streaming inference prediction.
    """
    add_trace_resource_name("task_streaming_post")
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
                        status=TaskStatus.FAILURE, traceback=exc.content.decode()
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
