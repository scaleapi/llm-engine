from model_engine_server.common.dtos.files import (
    DeleteFileResponse,
    GetFileContentResponse,
    GetFileResponse,
    ListFilesResponse,
    UploadFileResponse,
)
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.domain.exceptions import ObjectNotFoundException
from model_engine_server.core.loggers import filename_wo_ext, make_logger
from model_engine_server.domain.gateways import FileStorageGateway

logger = make_logger(filename_wo_ext(__file__))


class UploadFileUseCase:
    def __init__(self, file_storage_gateway: FileStorageGateway):
        self.file_storage_gateway = file_storage_gateway

    async def execute(self, user: User, filename: str, content: bytes) -> UploadFileResponse:
        file_id = await self.file_storage_gateway.upload_file(
            owner=user.team_id,
            filename=filename,
            content=content,
        )
        return UploadFileResponse(
            id=file_id,
        )


class GetFileUseCase:
    def __init__(self, file_storage_gateway: FileStorageGateway):
        self.file_storage_gateway = file_storage_gateway

    async def execute(self, user: User, file_id: str) -> GetFileResponse:
        file_metadata = await self.file_storage_gateway.get_file(
            owner=user.team_id,
            file_id=file_id,
        )
        if file_metadata is None:
            raise ObjectNotFoundException
        return GetFileResponse(
            id=file_metadata.id,
            filename=file_metadata.filename,
            size=file_metadata.size,
        )


class ListFilesUseCase:
    def __init__(self, file_storage_gateway: FileStorageGateway):
        self.file_storage_gateway = file_storage_gateway

    async def execute(self, user: User) -> ListFilesResponse:
        files = await self.file_storage_gateway.list_files(
            owner=user.team_id,
        )
        return ListFilesResponse(
            files=[
                GetFileResponse(
                    id=file_metadata.id,
                    filename=file_metadata.filename,
                    size=file_metadata.size,
                )
                for file_metadata in files
            ]
        )


class DeleteFileUseCase:
    def __init__(self, file_storage_gateway: FileStorageGateway):
        self.file_storage_gateway = file_storage_gateway

    async def execute(self, user: User, file_id: str) -> DeleteFileResponse:
        deleted = await self.file_storage_gateway.delete_file(
            owner=user.team_id,
            file_id=file_id,
        )
        return DeleteFileResponse(
            deleted=deleted,
        )


class GetFileContentUseCase:
    def __init__(self, file_storage_gateway: FileStorageGateway):
        self.file_storage_gateway = file_storage_gateway

    async def execute(self, user: User, file_id: str) -> GetFileContentResponse:
        content = await self.file_storage_gateway.get_file_content(
            owner=user.team_id,
            file_id=file_id,
        )
        if content is None:
            raise ObjectNotFoundException
        return GetFileContentResponse(
            id=file_id,
            content=content,
        )
