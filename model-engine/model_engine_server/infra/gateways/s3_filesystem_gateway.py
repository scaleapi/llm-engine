import os
import re
from typing import IO

import boto3
import smart_open
from model_engine_server.core.config import infra_config
from model_engine_server.infra.gateways.filesystem_gateway import FilesystemGateway


class S3FilesystemGateway(FilesystemGateway):
    """
    Concrete implementation for interacting with a filesystem backed by S3.
    """

    def get_s3_client(self, kwargs):
        if infra_config().cloud_provider == "onprem":
            # For onprem, use explicit credentials from environment variables
            session = boto3.Session(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=infra_config().default_region
            )
        else:
            profile_name = kwargs.get("aws_profile", os.getenv("AWS_PROFILE"))
            session = boto3.Session(profile_name=profile_name)
        
        # Support custom endpoints for S3-compatible storage (like Scality)
        # Uses standard boto3 environment variables
        endpoint_url = kwargs.get("endpoint_url") or os.getenv("AWS_ENDPOINT_URL")
        
        client_kwargs = {}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
            # For custom endpoints, boto3 automatically uses AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
        
        client = session.client("s3", **client_kwargs)
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
