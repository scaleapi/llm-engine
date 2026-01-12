import os
from typing import List, Optional

from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.gateways.file_storage_gateway import (
    FileMetadata,
    FileStorageGateway,
)
from model_engine_server.infra.gateways.s3_filesystem_gateway import S3FilesystemGateway

logger = make_logger(logger_name())


def get_s3_key(owner: str, file_id: str) -> str:
    return os.path.join(owner, file_id)


def get_s3_url(owner: str, file_id: str) -> str:
    return f"s3://{infra_config().s3_bucket}/{get_s3_key(owner, file_id)}"


class S3FileStorageGateway(FileStorageGateway):
    def __init__(self):
        self.filesystem_gateway = S3FilesystemGateway()

    async def get_url_from_id(self, owner: str, file_id: str) -> Optional[str]:
        try:
            url = self.filesystem_gateway.generate_signed_url(get_s3_url(owner, file_id))
            logger.debug(f"Generated presigned URL for {owner}/{file_id}")
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {owner}/{file_id}: {e}")
            return None

    async def get_file(self, owner: str, file_id: str) -> Optional[FileMetadata]:
        try:
            obj = self.filesystem_gateway.head_object(
                bucket=infra_config().s3_bucket,
                key=get_s3_key(owner, file_id),
            )
            return FileMetadata(
                id=file_id,
                filename=file_id,
                size=obj.get("ContentLength"),
                owner=owner,
                updated_at=obj.get("LastModified"),
            )
        except Exception as e:
            logger.debug(f"File not found or error retrieving {owner}/{file_id}: {e}")
            return None

    async def get_file_content(self, owner: str, file_id: str) -> Optional[str]:
        try:
            with self.filesystem_gateway.open(
                get_s3_url(owner, file_id), aws_profile=infra_config().profile_ml_worker
            ) as f:
                content = f.read()
            logger.debug(f"Retrieved content for {owner}/{file_id}")
            return content
        except Exception as e:
            logger.error(f"Failed to read file {owner}/{file_id}: {e}")
            return None

    async def upload_file(self, owner: str, filename: str, content: bytes) -> str:
        with self.filesystem_gateway.open(
            get_s3_url(owner, filename), mode="w", aws_profile=infra_config().profile_ml_worker
        ) as f:
            f.write(content.decode("utf-8"))
        logger.info(f"Uploaded file {owner}/{filename}")
        return filename

    async def delete_file(self, owner: str, file_id: str) -> bool:
        try:
            self.filesystem_gateway.delete_object(
                bucket=infra_config().s3_bucket,
                key=get_s3_key(owner, file_id),
            )
            logger.info(f"Deleted file {owner}/{file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {owner}/{file_id}: {e}")
            return False

    async def list_files(self, owner: str) -> List[FileMetadata]:
        try:
            objects = self.filesystem_gateway.list_objects(
                bucket=infra_config().s3_bucket,
                prefix=owner,
            )
            files = []
            for obj in objects:
                key = obj["Key"]
                if key.startswith(owner):
                    file_id = key[len(owner) :].lstrip("/")
                    if file_id:
                        file_metadata = await self.get_file(owner, file_id)
                        if file_metadata:
                            files.append(file_metadata)
            logger.debug(f"Listed {len(files)} files for owner {owner}")
            return files
        except Exception as e:
            logger.error(f"Failed to list files for owner {owner}: {e}")
            return []
