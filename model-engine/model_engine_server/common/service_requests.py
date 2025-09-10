# Functions to make requests to individual services / Servable instantiations

from typing import Any, Dict, Optional

import requests
from model_engine_server.common.errors import HTTP429Exception, UpstreamHTTPSvcError
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = make_logger(logger_name())

SYNC_ENDPOINT_RETRIES = 10  # Must be an integer >= 0
SYNC_ENDPOINT_MAX_TIMEOUT_SECONDS = 10


def make_sync_request_with_retries(
    request_url: str,
    payload_json: Dict[str, Any],
    timeout_seconds: Optional[int] = SYNC_ENDPOINT_MAX_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    # Copied from document-endpoint
    # More details at https://tenacity.readthedocs.io/en/latest/#retrying-code-block
    # Try/catch + for loop makes us retry only when we get a 429 from the synchronous endpoint.
    # We should be creating a new requests Session each time, which should avoid sending requests to the same endpoint
    # This is admittedly a hack until we get proper least-outstanding-requests load balancing to our http endpoints
    if infra_config().debug_mode:
        logger.info(f"DEBUG: make_sync_request_with_retries to URL: {request_url}")
        logger.info(
            f"DEBUG: Payload keys: {list(payload_json.keys()) if isinstance(payload_json, dict) else type(payload_json)}"
        )

    try:
        for attempt in Retrying(
            stop=stop_after_attempt(SYNC_ENDPOINT_RETRIES + 1),
            retry=retry_if_exception_type(HTTP429Exception),
            wait=wait_exponential(multiplier=1, min=1, max=timeout_seconds),
        ):
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    if infra_config().debug_mode:
                        logger.info(f"Retry number {attempt.retry_state.attempt_number}")

                if infra_config().debug_mode:
                    logger.info(
                        f"DEBUG: About to POST to {request_url} (attempt {attempt.retry_state.attempt_number})"
                    )

                try:
                    resp = requests.post(
                        request_url,
                        json=payload_json,
                        headers={"Content-Type": "application/json"},
                    )
                    if infra_config().debug_mode:
                        logger.info(f"DEBUG: Response status: {resp.status_code}")
                except Exception as e:
                    if infra_config().debug_mode:
                        logger.error(
                            f"DEBUG: Exception during requests.post: {type(e).__name__}: {e}"
                        )
                    raise

                if resp.status_code == 429:
                    raise HTTP429Exception("429 returned")
                elif resp.status_code != 200:
                    if infra_config().debug_mode:
                        logger.warning(
                            f"DEBUG: Non-200 response. Status: {resp.status_code}, Content: {resp.content}"
                        )
                    raise UpstreamHTTPSvcError(status_code=resp.status_code, content=resp.content)
                return resp.json()
    except RetryError:
        logger.warning("Hit max # of retries, returning 429 to client")
        raise UpstreamHTTPSvcError(status_code=429, content=resp.content)
    # Never reached because tenacity should throw a RetryError if we exit the for loop. This is for mypy.
    return resp.json()
