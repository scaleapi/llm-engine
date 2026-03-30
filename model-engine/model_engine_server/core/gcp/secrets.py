"""GCP secrets module."""

import json
from functools import lru_cache

from google.cloud import secretmanager
from model_engine_server.core.loggers import logger_name, make_logger

logger = make_logger(logger_name())


@lru_cache(maxsize=2)
def get_key_file(secret_name: str):
    """Fetch and parse a JSON secret from GCP Secret Manager.

    secret_name should be a fully-qualified resource name:
        projects/{project}/secrets/{secret}/versions/latest
    or a short name if the GCP project is resolvable from the environment.
    """
    client = secretmanager.SecretManagerServiceClient()
    try:
        response = client.access_secret_version(name=secret_name)
        return json.loads(response.payload.data.decode("utf-8"))
    except Exception as e:
        logger.error(e)
        logger.error(f"Failed to retrieve secret: {secret_name}")
        return {}
