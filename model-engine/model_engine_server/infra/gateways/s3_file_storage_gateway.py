import os
from typing import List, Optional

from model_engine_server.core.config import infra_config
from model_engine_server.domain.gateways.file_storage_gateway import (
    FileMetadata,
    FileStorageGateway,
)
from model_engine_server.infra.gateways import S3FilesystemGateway


def get_s3_key(owner: str, file_id: str):
    return os.path.join(owner, file_id)


def get_s3_url(owner: str, file_id: str):
    return f"s3://{infra_config().s3_bucket}/{get_s3_key(owner, file_id)}"


class S3FileStorageGateway(FileStorageGateway):
    """
    Concrete implementation of a file storage gateway backed by S3.
    """

    def __init__(self):
        self.filesystem_gateway = S3FilesystemGateway()

    async def get_url_from_id(self, owner: str, file_id: str) -> Optional[str]:
        return self.filesystem_gateway.generate_signed_url(get_s3_url(owner, file_id))

    async def get_file(self, owner: str, file_id: str) -> Optional[FileMetadata]:
        try:
            obj = self.filesystem_gateway.get_s3_client({}).head_object(
                Bucket=infra_config().s3_bucket,
                Key=get_s3_key(owner, file_id),
            )
            return FileMetadata(
                id=file_id,
                filename=file_id,
                size=obj.get("ContentLength"),
                owner=owner,
                updated_at=obj.get("LastModified"),
            )
        except:  # noqa: E722
            return None

    async def get_file_content(self, owner: str, file_id: str) -> Optional[str]:
        try:
            with self.filesystem_gateway.open(
                get_s3_url(owner, file_id), aws_profile=infra_config().profile_ml_worker
            ) as f:
                return f.read()
        except:  # noqa: E722
            return None

    async def upload_file(self, owner: str, filename: str, content: bytes) -> str:
        with self.filesystem_gateway.open(
            get_s3_url(owner, filename), mode="w", aws_profile=infra_config().profile_ml_worker
        ) as f:
            f.write(content.decode("utf-8"))
        return filename

    async def delete_file(self, owner: str, file_id: str) -> bool:
        try:
            self.filesystem_gateway.get_s3_client({}).delete_object(
                Bucket=infra_config().s3_bucket,
                Key=get_s3_key(owner, file_id),
            )
            return True
        except:  # noqa: E722
            return False

    async def list_files(self, owner: str) -> List[FileMetadata]:
        objects = self.filesystem_gateway.get_s3_client({}).list_objects_v2(
            Bucket=infra_config().s3_bucket,
            Prefix=owner,
        )
        files = [await self.get_file(owner, obj["Name"]) for obj in objects]
        return [f for f in files if f is not None]
