"""GCP secrets module."""

import json
from functools import lru_cache
from typing import Optional

from google.cloud import secretmanager
from model_engine_server.core.loggers import logger_name, make_logger

logger = make_logger(logger_name())


@lru_cache(maxsize=2)
def get_key_file(secret_name: str, gcp_project: Optional[str] = None):
    """Fetch and parse a JSON secret from GCP Secret Manager.

    If gcp_project is provided, secret_name is treated as a short name and
    the full resource name is constructed as:
        projects/{gcp_project}/secrets/{secret_name}/versions/latest

    Otherwise, secret_name should already be a fully-qualified resource name.
    """
    if gcp_project is not None:
        secret_name = f"projects/{gcp_project}/secrets/{secret_name}/versions/latest"
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(name=secret_name)
    return json.loads(response.payload.data.decode("utf-8"))
