import re
from typing import IO

import smart_open
from model_engine_server.infra.gateways.filesystem_gateway import FilesystemGateway
from model_engine_server.infra.gateways.s3_utils import get_s3_client


class S3FilesystemGateway(FilesystemGateway):
    def open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        client = get_s3_client(kwargs)
        transport_params = {"client": client}
        return smart_open.open(uri, mode, transport_params=transport_params)

    def generate_signed_url(self, uri: str, expiration: int = 3600, **kwargs) -> str:
        client = get_s3_client(kwargs)
        match = re.search(r"^s3://([^/]+)/(.*?)$", uri)
        if not match:
            raise ValueError(f"Invalid S3 URI format: {uri}")

        bucket, key = match.group(1), match.group(2)
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key, "ResponseContentType": "text/plain"},
            ExpiresIn=expiration,
        )
