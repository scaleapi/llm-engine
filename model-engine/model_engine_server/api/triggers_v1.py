from fastapi import APIRouter, Depends, HTTPException
from model_engine_server.api.dependencies import (
    ExternalInterfaces,
    get_external_interfaces,
    verify_authentication,
)
from model_engine_server.common.datadog_utils import add_trace_resource_name
from model_engine_server.common.dtos.triggers import (
    CreateTriggerV1Request,
    CreateTriggerV1Response,
    DeleteTriggerV1Response,
    GetTriggerV1Response,
    ListTriggersV1Response,
    UpdateTriggerV1Request,
    UpdateTriggerV1Response,
)
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import (
    CronSyntaxException,
    DockerImageNotFoundException,
    EndpointLabelsException,
    EndpointResourceInvalidRequestException,
    ObjectHasInvalidValueException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
    TriggerNameAlreadyExistsException,
)
from model_engine_server.domain.use_cases.trigger_use_cases import (
    CreateTriggerUseCase,
    DeleteTriggerUseCase,
    GetTriggerUseCase,
    ListTriggersUseCase,
    UpdateTriggerUseCase,
)

trigger_router_v1 = APIRouter(prefix="/v1")

logger = make_logger(logger_name())


@trigger_router_v1.post("/triggers", response_model=CreateTriggerV1Response)
async def create_trigger(
    request: CreateTriggerV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> CreateTriggerV1Response:
    """
    Creates and runs a trigger
    """
    add_trace_resource_name("triggers_post")
    logger.info(f"POST /triggers with {request} for {auth}")
    try:
        use_case = CreateTriggerUseCase(
            trigger_repository=external_interfaces.trigger_repository,
            cron_job_gateway=external_interfaces.cron_job_gateway,
            docker_image_batch_job_bundle_repository=external_interfaces.docker_image_batch_job_bundle_repository,
            docker_repository=external_interfaces.docker_repository,
        )
        return await use_case.execute(user=auth, request=request)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404, detail="The specified batch job bundle could not be found"
        ) from exc
    except DockerImageNotFoundException as exc:
        raise HTTPException(
            status_code=404,
            detail=f"The specified docker image {exc.repository}:{exc.tag} was not found",
        )
    except ObjectHasInvalidValueException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"The user specified an invalid value: {exc}",
        ) from exc
    except EndpointResourceInvalidRequestException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Default trigger resource request is invalid: {exc}",
        )
    except EndpointLabelsException as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except CronSyntaxException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"The user specified an invalid value for cron_schedule: {exc}",
        )
    except TriggerNameAlreadyExistsException as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@trigger_router_v1.get("/triggers", response_model=ListTriggersV1Response)
async def list_triggers(
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> ListTriggersV1Response:
    """
    Lists descriptions of all triggers
    """
    add_trace_resource_name("triggers_get")
    logger.info(f"GET /triggers for {auth}")
    use_case = ListTriggersUseCase(trigger_repository=external_interfaces.trigger_repository)
    return await use_case.execute(user=auth)


@trigger_router_v1.get("/triggers/{trigger_id}", response_model=GetTriggerV1Response)
async def get_trigger(
    trigger_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> GetTriggerV1Response:
    """
    Describes the trigger with the given ID
    """
    add_trace_resource_name("triggers_id_get")
    logger.info(f"GET /triggers/{trigger_id} for {auth}")
    try:
        use_case = GetTriggerUseCase(trigger_repository=external_interfaces.trigger_repository)
        return await use_case.execute(user=auth, trigger_id=trigger_id)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(status_code=404, detail=f"Trigger {trigger_id} was not found.") from exc


@trigger_router_v1.put("/triggers/{trigger_id}", response_model=UpdateTriggerV1Response)
async def update_trigger(
    trigger_id: str,
    request: UpdateTriggerV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> UpdateTriggerV1Response:
    """
    Updates the trigger with the given ID
    """
    add_trace_resource_name("triggers_id_put")
    logger.info(f"PUT /triggers/{trigger_id} with {request} for {auth}")
    try:
        use_case = UpdateTriggerUseCase(
            trigger_repository=external_interfaces.trigger_repository,
            cron_job_gateway=external_interfaces.cron_job_gateway,
        )
        return await use_case.execute(user=auth, trigger_id=trigger_id, request=request)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(status_code=404, detail=f"Trigger {trigger_id} was not found.") from exc
    except CronSyntaxException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"The user specified an invalid value for cron_schedule: {exc}",
        )


@trigger_router_v1.delete("/triggers/{trigger_id}", response_model=DeleteTriggerV1Response)
async def delete_trigger(
    trigger_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> DeleteTriggerV1Response:
    """
    Deletes the trigger with the given ID
    """
    add_trace_resource_name("trigger_id_delete")
    logger.info(f"DELETE /triggers/{trigger_id} for {auth}")
    try:
        use_case = DeleteTriggerUseCase(
            trigger_repository=external_interfaces.trigger_repository,
            cron_job_gateway=external_interfaces.cron_job_gateway,
        )
        return await use_case.execute(user=auth, trigger_id=trigger_id)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(status_code=404, detail=f"Trigger {trigger_id} was not found.") from exc
