import json
from typing import Dict


# LLM Engine Errors
class ValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class UnauthorizedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


# API Inference Errors
class BadRequestError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class NotFoundError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class RateLimitExceededError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


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

    # Fallback to an unknown error
    return UnknownError(message)
