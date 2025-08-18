"""AWS secrets module."""

import json
import os
from functools import lru_cache
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger

logger = make_logger(logger_name())


@lru_cache(maxsize=2)
def get_key_file(secret_name: str, aws_profile: Optional[str] = None):
    # Check if AWS Secrets Manager is disabled via config
    if infra_config().disable_aws_secrets_manager:
        logger.warning(f"AWS Secrets Manager disabled - cannot retrieve secret: {secret_name}")
        return {}
    
    if aws_profile is not None:
        session = boto3.Session(profile_name=aws_profile)
        secret_manager = session.client("secretsmanager", region_name=infra_config().default_region)
    else:
        secret_manager = boto3.client("secretsmanager", region_name=infra_config().default_region)
    response = secret_manager.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])
