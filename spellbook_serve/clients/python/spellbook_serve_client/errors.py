from typing import Dict


# Spellbook Serve Errors
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


def parse_error(status_code: int, payload: Dict[str, str]) -> Exception:
    """
    Parse error given an HTTP status code and a json payload

    Args:
        status_code (`int`):
            HTTP status code
        payload (`Dict[str, str]`):
            Json payload

    Returns:
        Exception: parsed exception

    """
    # Try to parse a Spellbook Serve error
    message = payload["detail"]

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
