"""AWS secrets module."""
import json
from functools import lru_cache
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from llm_engine_server.core.config import ml_infra_config
from llm_engine_server.core.loggers import filename_wo_ext, make_logger

logger = make_logger(filename_wo_ext(__file__))


@lru_cache(maxsize=2)
def get_key_file(secret_name: str, aws_profile: Optional[str] = None):
    if aws_profile is not None:
        session = boto3.Session(profile_name=aws_profile)
        secret_manager = session.client(
            "secretsmanager", region_name=ml_infra_config().default_region
        )
    else:
        secret_manager = boto3.client(
            "secretsmanager", region_name=ml_infra_config().default_region
        )
    try:
        secret_value = json.loads(
            secret_manager.get_secret_value(SecretId=secret_name)["SecretString"]
        )
        return secret_value
    except ClientError as e:
        logger.error(e)
        logger.error(f"Failed to retrieve secret: {secret_name}")
        return {}
