"""This module provides a client for the AWS Step Functions service."""
import os
from typing import Optional

from botocore.client import BaseClient

from llm_engine_server.core.aws.roles import session
from llm_engine_server.core.config import ml_infra_config
from llm_engine_server.core.loggers import logger_name, make_logger

logger = make_logger(logger_name())


def sync_sfn_client(**kwargs) -> Optional[BaseClient]:
    is_testing_mode = os.environ.get("TESTING_DISABLE_SFN", "").lower() == "true"
    if is_testing_mode:
        logger.error(
            "Not creating step function client as we are in testing mode."
            "THIS SHOULD NOT HAPPEN IN PRODUCTION!"
        )
        return None
    return session(ml_infra_config().profile_ml_worker).client("stepfunctions", **kwargs)
