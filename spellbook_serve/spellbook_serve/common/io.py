"""Launch Input/Output utils."""
import os

import boto3
import smart_open


def open_wrapper(uri: str, mode: str = "rt", **kwargs):
    # This follows the 5.1.0 smart_open API
    profile_name = kwargs.get("aws_profile", os.getenv("AWS_PROFILE"))
    session = boto3.Session(profile_name=profile_name)
    client = session.client("s3")
    transport_params = {"client": client}
    return smart_open.open(uri, mode, transport_params=transport_params)
