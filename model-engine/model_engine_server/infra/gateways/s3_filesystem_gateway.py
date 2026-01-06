import os
import re
from typing import IO

import boto3
import smart_open
from model_engine_server.infra.gateways.filesystem_gateway import FilesystemGateway


class S3FilesystemGateway(FilesystemGateway):
    """
    Concrete implementation for interacting with a filesystem backed by S3.
    """

    def get_s3_client(self, kwargs):
        profile_name = kwargs.get("aws_profile", os.getenv("AWS_PROFILE"))
        # For on-prem: if profile_name is empty/None, use default credential chain (env vars)
        if profile_name:
            session = boto3.Session(profile_name=profile_name)
        else:
            session = boto3.Session()
        
        # Support for MinIO/on-prem S3-compatible storage
        endpoint_url = os.getenv("S3_ENDPOINT_URL")
        client = session.client("s3", endpoint_url=endpoint_url)
        return client

    def open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        # This follows the 5.1.0 smart_open API
        client = self.get_s3_client(kwargs)
        transport_params = {"client": client}
        return smart_open.open(uri, mode, transport_params=transport_params)

    def generate_signed_url(self, uri: str, expiration: int = 3600, **kwargs) -> str:
        client = self.get_s3_client(kwargs)
        match = re.search("^s3://([^/]+)/(.*?)$", uri)
        assert match

        bucket, key = match.group(1), match.group(2)
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key, "ResponseContentType": "text/plain"},
            ExpiresIn=expiration,
        )
