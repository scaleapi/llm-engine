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
    # Only use AWS Secrets Manager for AWS cloud provider
    if infra_config().cloud_provider != "aws":
        logger.warning(f"Not using AWS Secrets Manager - cloud provider is {infra_config().cloud_provider} (cannot retrieve secret: {secret_name})")
        return {}
    
    try:
        if aws_profile is not None:
            session = boto3.Session(profile_name=aws_profile)
            secret_manager = session.client("secretsmanager", region_name=infra_config().default_region)
        else:
            secret_manager = boto3.client("secretsmanager", region_name=infra_config().default_region)
        
        response = secret_manager.get_secret_value(SecretId=secret_name)
        return json.loads(response["SecretString"])
    except (ClientError, Exception) as e:
        logger.warning(f"Failed to retrieve secret {secret_name} from AWS Secrets Manager: {e}")
        return {}
