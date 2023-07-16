import json


# LLM Engine Errors
class ValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


# API Inference Errors
class BadRequestError(Exception):
    """
    Corresponds to HTTP 400. Indicates that the request had inputs that were invalid. The user should not
    attempt to retry the request without changing the inputs.
    """

    def __init__(self, message: str):
        super().__init__(message)


class UnauthorizedError(Exception):
    """
    Corresponds to HTTP 401. This means that no valid API key was provided.
    """

    def __init__(self, message: str):
        super().__init__(message)


class NotFoundError(Exception):
    """
    Corresponds to HTTP 404. This means that the resource (e.g. a Model, FineTune, etc.) could not be found.
    Note that this can also be returned in some cases where the object might exist, but the user does not have access
    to the object. This is done to avoid leaking information about the existence or nonexistence of said object that
    the user does not have access to.
    """

    def __init__(self, message: str):
        super().__init__(message)


class RateLimitExceededError(Exception):
    """
    Corresponds to HTTP 429. Too many requests hit the API too quickly. We recommend an exponential backoff for retries.
    """

    def __init__(self, message: str):
        super().__init__(message)


class ServerError(Exception):
    """
    Corresponds to HTTP 5xx errors on the server.
    """

    def __init__(self, status_code: int, message: str):
        super().__init__(f"Server exception with {status_code=}, {message=}")


# Unknown error
class UnknownError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def parse_error(status_code: int, content: bytes) -> Exception:
    """
    Parse error given an HTTP status code and a bytes payload

    Args:
        status_code (`int`):
            HTTP status code
        content (`bytes`):
            payload

    Returns:
        Exception: parsed exception

    """
    # Try to parse a LLM Engine error
    try:
        payload = json.loads(content)
        message = payload["detail"]
    except json.JSONDecodeError:
        message = content.decode("utf-8")

    # Try to parse a APIInference error
    if status_code == 400:
        return BadRequestError(message)
    if status_code == 401:
        return UnauthorizedError(message)
    if status_code == 404:
        return NotFoundError(message)
    if status_code == 429:
        return RateLimitExceededError(message)
    if 600 < status_code <= 500:
        return ServerError(status_code, message)

    # Fallback to an unknown error
    return UnknownError(message)
