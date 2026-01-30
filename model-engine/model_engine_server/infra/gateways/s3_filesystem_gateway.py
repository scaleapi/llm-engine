import re
from typing import IO, Any, Dict, List, Optional

import smart_open
from model_engine_server.infra.gateways.filesystem_gateway import FilesystemGateway
from model_engine_server.infra.gateways.s3_utils import get_s3_client


class S3FilesystemGateway(FilesystemGateway):
    def _get_client(self, kwargs: Optional[Dict[str, Any]] = None) -> Any:
        return get_s3_client(kwargs or {})

    def open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        client = self._get_client(kwargs)
        transport_params = {"client": client}
        return smart_open.open(uri, mode, transport_params=transport_params)

    def generate_signed_url(self, uri: str, expiration: int = 3600, **kwargs) -> str:
        client = self._get_client(kwargs)
        match = re.search(r"^s3://([^/]+)/(.*?)$", uri)
        if not match:
            raise ValueError(f"Invalid S3 URI format: {uri}")

        bucket, key = match.group(1), match.group(2)
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key, "ResponseContentType": "text/plain"},
            ExpiresIn=expiration,
        )

    def head_object(self, bucket: str, key: str, **kwargs) -> Dict[str, Any]:
        client = self._get_client(kwargs)
        return client.head_object(Bucket=bucket, Key=key)

    def delete_object(self, bucket: str, key: str, **kwargs) -> Dict[str, Any]:
        client = self._get_client(kwargs)
        return client.delete_object(Bucket=bucket, Key=key)

    def list_objects(self, bucket: str, prefix: str, **kwargs) -> List[Dict[str, Any]]:
        client = self._get_client(kwargs)
        paginator = client.get_paginator("list_objects_v2")
        contents: List[Dict[str, Any]] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            contents.extend(page.get("Contents", []))
        return contents
