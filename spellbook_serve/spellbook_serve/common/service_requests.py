# Functions to make requests to individual services / Servable instantiations

from typing import Any, Dict, Optional

import requests
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from spellbook_serve.common.errors import HTTP429Exception, UpstreamHTTPSvcError
from spellbook_serve.core.loggers import filename_wo_ext, make_logger

logger = make_logger(filename_wo_ext(__file__))

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

    try:
        for attempt in Retrying(
            stop=stop_after_attempt(SYNC_ENDPOINT_RETRIES + 1),
            retry=retry_if_exception_type(HTTP429Exception),
            wait=wait_exponential(multiplier=1, min=1, max=timeout_seconds),
        ):
            with attempt:
                logger.info(f"Retry number {attempt.retry_state.attempt_number}")
                resp = requests.post(
                    request_url,
                    json=payload_json,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 429:
                    raise HTTP429Exception("429 returned")
                elif resp.status_code != 200:
                    raise UpstreamHTTPSvcError(status_code=resp.status_code, content=resp.content)
                return resp.json()
    except RetryError:
        logger.warning("Hit max # of retries, returning 429 to client")
        raise UpstreamHTTPSvcError(status_code=429, content=resp.content)
    # Never reached because tenacity should throw a RetryError if we exit the for loop. This is for mypy.
    return resp.json()
