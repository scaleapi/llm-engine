"""LLM Model Endpoint routes for the hosted model inference service.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from llm_engine_server.api.dependencies import (
    ExternalInterfaces,
    get_external_interfaces,
    get_external_interfaces_read_only,
    verify_authentication,
)
from llm_engine_server.common.datadog_utils import add_trace_resource_name
from llm_engine_server.common.dtos.llms import (
    CancelFineTuneJobResponse,
    CompletionStreamV1Request,
    CompletionStreamV1Response,
    CompletionSyncV1Request,
    CompletionSyncV1Response,
    CreateFineTuneJobRequest,
    CreateFineTuneJobResponse,
    CreateLLMModelEndpointV1Request,
    CreateLLMModelEndpointV1Response,
    GetFineTuneJobResponse,
    GetLLMModelEndpointV1Response,
    ListFineTuneJobResponse,
    ListLLMModelEndpointsV1Response,
)
from llm_engine_server.common.dtos.model_endpoints import ModelEndpointOrderBy
from llm_engine_server.common.dtos.tasks import TaskStatus
from llm_engine_server.core.auth.authentication_repository import User
from llm_engine_server.core.domain_exceptions import (
    ObjectAlreadyExistsException,
    ObjectHasInvalidValueException,
    ObjectNotApprovedException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from llm_engine_server.core.loggers import filename_wo_ext, make_logger
from llm_engine_server.domain.exceptions import (
    EndpointLabelsException,
    EndpointResourceInvalidRequestException,
    EndpointUnsupportedInferenceTypeException,
    InvalidRequestException,
    LLMFineTuningMethodNotImplementedException,
    UpstreamServiceError,
)
from llm_engine_server.domain.use_cases.llm_fine_tuning_use_cases import (
    CancelFineTuneJobV1UseCase,
    CreateFineTuneJobV1UseCase,
    GetFineTuneJobV1UseCase,
    ListFineTuneJobV1UseCase,
)
from llm_engine_server.domain.use_cases.llm_model_endpoint_use_cases import (
    CompletionStreamV1UseCase,
    CompletionSyncV1UseCase,
    CreateLLMModelEndpointV1UseCase,
    GetLLMModelEndpointByNameV1UseCase,
    ListLLMModelEndpointsV1UseCase,
)
from llm_engine_server.domain.use_cases.model_bundle_use_cases import CreateModelBundleV2UseCase
from sse_starlette.sse import EventSourceResponse

llm_router_v1 = APIRouter(prefix="/v1/llm")
logger = make_logger(filename_wo_ext(__name__))


@llm_router_v1.post("/model-endpoints", response_model=CreateLLMModelEndpointV1Response)
async def create_model_endpoint(
    request: CreateLLMModelEndpointV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> CreateLLMModelEndpointV1Response:
    """
    Creates an LLM endpoint for the current user.
    """
    add_trace_resource_name("llm_model_endpoints_post")
    logger.info(f"POST /llm/model-endpoints with {request} for {auth}")
    try:
        create_model_bundle_use_case = CreateModelBundleV2UseCase(
            model_bundle_repository=external_interfaces.model_bundle_repository,
            docker_repository=external_interfaces.docker_repository,
            model_primitive_gateway=external_interfaces.model_primitive_gateway,
        )
        use_case = CreateLLMModelEndpointV1UseCase(
            create_model_bundle_use_case=create_model_bundle_use_case,
            model_bundle_repository=external_interfaces.model_bundle_repository,
            model_endpoint_service=external_interfaces.model_endpoint_service,
        )
        return await use_case.execute(user=auth, request=request)
    except ObjectAlreadyExistsException as exc:
        raise HTTPException(
            status_code=400,
            detail="The specified model endpoint already exists.",
        ) from exc
    except EndpointLabelsException as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except ObjectHasInvalidValueException as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except EndpointResourceInvalidRequestException as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except ObjectNotApprovedException as exc:
        raise HTTPException(
            status_code=403,
            detail="The specified model bundle was not approved yet.",
        ) from exc
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified model bundle could not be found.",
        ) from exc


@llm_router_v1.get("/model-endpoints", response_model=ListLLMModelEndpointsV1Response)
async def list_model_endpoints(
    name: Optional[str] = Query(default=None),
    order_by: Optional[ModelEndpointOrderBy] = Query(default=None),
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> ListLLMModelEndpointsV1Response:
    """
    Lists the LLM model endpoints owned by the current owner, plus all public_inference LLMs.
    """
    add_trace_resource_name("llm_model_endpoints_get")
    logger.info(f"GET /llm/model-endpoints?name={name}&order_by={order_by} for {auth}")
    use_case = ListLLMModelEndpointsV1UseCase(
        llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service,
    )
    return await use_case.execute(user=auth, name=name, order_by=order_by)


@llm_router_v1.get(
    "/model-endpoints/{model_endpoint_name}",
    response_model=GetLLMModelEndpointV1Response,
)
async def get_model_endpoint(
    model_endpoint_name: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> GetLLMModelEndpointV1Response:
    """
    Describe the LLM Model endpoint with given name.
    """
    add_trace_resource_name("llm_model_endpoints_name_get")
    logger.info(f"GET /llm/model-endpoints/{model_endpoint_name} for {auth}")
    try:
        use_case = GetLLMModelEndpointByNameV1UseCase(
            llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service
        )
        return await use_case.execute(user=auth, model_endpoint_name=model_endpoint_name)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Model Endpoint {model_endpoint_name}  was not found.",
        ) from exc


@llm_router_v1.post("/completions-sync", response_model=CompletionSyncV1Response)
async def create_completion_sync_task(
    model_endpoint_name: str,
    request: CompletionSyncV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> CompletionSyncV1Response:
    """
    Runs a sync prompt completion on an LLM.
    """
    add_trace_resource_name("llm_completion_sync_post")
    logger.info(
        f"POST /completion_sync with {request} to endpoint {model_endpoint_name} for {auth}"
    )
    try:
        use_case = CompletionSyncV1UseCase(
            model_endpoint_service=external_interfaces.model_endpoint_service,
            llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service,
        )
        return await use_case.execute(
            user=auth, model_endpoint_name=model_endpoint_name, request=request
        )
    except UpstreamServiceError as exc:
        return CompletionSyncV1Response(
            status=TaskStatus.FAILURE, outputs=[], traceback=exc.content.decode()
        )
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified endpoint could not be found.",
        ) from exc
    except ObjectHasInvalidValueException as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except EndpointUnsupportedInferenceTypeException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported inference type: {str(exc)}",
        ) from exc


@llm_router_v1.post("/completions-stream", response_model=CompletionStreamV1Response)
async def create_completion_stream_task(
    model_endpoint_name: str,
    request: CompletionStreamV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> EventSourceResponse:
    """
    Runs a stream prompt completion on an LLM.
    """
    add_trace_resource_name("llm_completion_stream_post")
    logger.info(
        f"POST /completion_stream with {request} to endpoint {model_endpoint_name} for {auth}"
    )
    try:
        use_case = CompletionStreamV1UseCase(
            model_endpoint_service=external_interfaces.model_endpoint_service,
            llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service,
        )
        response = use_case.execute(
            user=auth, model_endpoint_name=model_endpoint_name, request=request
        )

        async def event_generator():
            async for message in response:
                yield {"data": message.json()}

        return EventSourceResponse(event_generator())
    except UpstreamServiceError as exc:
        return EventSourceResponse(
            iter(
                (
                    CompletionStreamV1Response(
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
    except ObjectHasInvalidValueException as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except EndpointUnsupportedInferenceTypeException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported inference type: {str(exc)}",
        ) from exc


@llm_router_v1.post("/fine-tunes", response_model=CreateFineTuneJobResponse)
async def create_fine_tune_job(
    request: CreateFineTuneJobRequest,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> CreateFineTuneJobResponse:
    add_trace_resource_name("fine_tunes_create")
    logger.info(f"POST /fine-tunes with {request} for {auth}")
    try:
        use_case = CreateFineTuneJobV1UseCase(
            llm_fine_tuning_service=external_interfaces.llm_fine_tuning_service,
        )
        return await use_case.execute(user=auth, request=request)
    except (LLMFineTuningMethodNotImplementedException, InvalidRequestException) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@llm_router_v1.get("/fine-tunes/{fine_tune_id}", response_model=GetFineTuneJobResponse)
async def get_fine_tune_job(
    fine_tune_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> GetFineTuneJobResponse:
    add_trace_resource_name("fine_tunes_get")
    logger.info(f"GET /fine-tunes/{fine_tune_id} for {auth}")
    try:
        use_case = GetFineTuneJobV1UseCase(
            llm_fine_tuning_service=external_interfaces.llm_fine_tuning_service,
        )
        return await use_case.execute(user=auth, fine_tune_id=fine_tune_id)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified fine-tune job could not be found.",
        ) from exc


@llm_router_v1.get("/fine-tunes", response_model=ListFineTuneJobResponse)
async def list_fine_tune_jobs(
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> ListFineTuneJobResponse:
    add_trace_resource_name("fine_tunes_list")
    logger.info(f"GET /fine-tunes for {auth}")
    use_case = ListFineTuneJobV1UseCase(
        llm_fine_tuning_service=external_interfaces.llm_fine_tuning_service,
    )
    return await use_case.execute(user=auth)


@llm_router_v1.put("/fine-tunes/{fine_tune_id}/cancel", response_model=CancelFineTuneJobResponse)
async def cancel_fine_tune_job(
    fine_tune_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> CancelFineTuneJobResponse:
    add_trace_resource_name("fine_tunes_cancel")
    logger.info(f"PUT /fine-tunes/{fine_tune_id}/cancel for {auth}")
    try:
        use_case = CancelFineTuneJobV1UseCase(
            llm_fine_tuning_service=external_interfaces.llm_fine_tuning_service,
        )
        return await use_case.execute(user=auth, fine_tune_id=fine_tune_id)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified fine-tune job could not be found.",
        ) from exc
