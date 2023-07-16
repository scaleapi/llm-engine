import os
import re
from typing import IO

import boto3
import smart_open

from . import FilesystemGateway


class S3FilesystemGateway(FilesystemGateway):
    """
    Concrete implemention for interacting with a filesystem backed by S3.
    """

    def _get_s3_client(self, kwargs):
        profile_name = kwargs.get("aws_profile", os.getenv("AWS_PROFILE"))
        session = boto3.Session(profile_name=profile_name)
        client = session.client("s3")
        return client

    def open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        # This follows the 5.1.0 smart_open API
        client = self._get_s3_client(kwargs)
        transport_params = {"client": client}
        return smart_open.open(uri, mode, transport_params=transport_params)

    def generate_signed_url(self, uri: str, expiration: int = 3600, **kwargs) -> str:
        client = self._get_s3_client(kwargs)
        match = re.search("^s3://([^/]+)/(.*?)$", uri)
        assert match

        bucket, key = match.group(1), match.group(2)
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key, "ResponseContentType": "text/plain"},
            ExpiresIn=expiration,
        )
