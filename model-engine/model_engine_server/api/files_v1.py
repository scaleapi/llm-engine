"""Files API routes for the hosted model inference service."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from model_engine_server.api.dependencies import (
    ExternalInterfaces,
    get_external_interfaces,
    get_external_interfaces_read_only,
    verify_authentication,
)
from model_engine_server.common.datadog_utils import add_trace_resource_name
from model_engine_server.common.dtos.files import (
    DeleteFileResponse,
    GetFileContentResponse,
    GetFileResponse,
    ListFilesResponse,
    UploadFileResponse,
)
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.loggers import filename_wo_ext, make_logger
from model_engine_server.domain.exceptions import (
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from model_engine_server.domain.use_cases.file_use_cases import (
    DeleteFileUseCase,
    GetFileContentUseCase,
    GetFileUseCase,
    ListFilesUseCase,
    UploadFileUseCase,
)

file_router_v1 = APIRouter(prefix="/v1")
logger = make_logger(filename_wo_ext(__name__))


@file_router_v1.post("/files", response_model=UploadFileResponse)
async def upload_file(
    file: UploadFile,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> UploadFileResponse:
    add_trace_resource_name("files_upload")
    logger.info(f"POST /files with filename {file.filename} for {auth}")
    use_case = UploadFileUseCase(
        file_storage_gateway=external_interfaces.file_storage_gateway,
    )
    return await use_case.execute(
        user=auth,
        filename=file.filename,
        content=file.file.read(),
    )


@file_router_v1.get("/files/{file_id}", response_model=GetFileResponse)
async def get_file(
    file_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> GetFileResponse:
    add_trace_resource_name("files_get")
    logger.info(f"GET /files/{file_id} for {auth}")
    try:
        use_case = GetFileUseCase(
            file_storage_gateway=external_interfaces.file_storage_gateway,
        )
        return await use_case.execute(user=auth, file_id=file_id)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified file could not be found.",
        ) from exc


@file_router_v1.get("/files", response_model=ListFilesResponse)
async def list_files(
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> ListFilesResponse:
    add_trace_resource_name("files_list")
    logger.info(f"GET /files for {auth}")
    use_case = ListFilesUseCase(
        file_storage_gateway=external_interfaces.file_storage_gateway,
    )
    return await use_case.execute(user=auth)


@file_router_v1.delete("/files/{file_id}", response_model=DeleteFileResponse)
async def delete_file(
    file_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> DeleteFileResponse:
    add_trace_resource_name("files_delete")
    logger.info(f"DELETE /files/{file_id} for {auth}")
    try:
        use_case = DeleteFileUseCase(
            file_storage_gateway=external_interfaces.file_storage_gateway,
        )
        return await use_case.execute(user=auth, file_id=file_id)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified file could not be found.",
        ) from exc


@file_router_v1.get("/files/{file_id}/content", response_model=GetFileContentResponse)
async def get_file_content(
    file_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> GetFileContentResponse:
    """
    Describe the LLM Model endpoint with given name.
    """
    add_trace_resource_name("files_content_get")
    logger.info(f"GET /files/{file_id}/content for {auth}")
    try:
        use_case = GetFileContentUseCase(
            file_storage_gateway=external_interfaces.file_storage_gateway,
        )
        return await use_case.execute(user=auth, file_id=file_id)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified file could not be found.",
        ) from exc
