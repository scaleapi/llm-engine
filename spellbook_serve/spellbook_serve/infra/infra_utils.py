# import logging
from logging import LoggerAdapter
from typing import Callable, Sequence

from spellbook_serve.common.env_vars import LOCAL

__all__: Sequence[str] = "make_exception_log"


def make_exception_log(logger_adapter: LoggerAdapter) -> Callable[[str], None]:
    """Makes a function to log error messages caused by exceptions.

    Iff LOCAL dev/test mode is enabled, then this returned function will log stacktraces.

    Always logs the error message (:param:emsg) to the `error` level on the
    :param:`logger_adapter`.
    """
    if LOCAL:
        _log_error: Callable[[str], None] = logger_adapter.exception
    else:
        _log_error = logger_adapter.error
    return _log_error


# def make_exception_log(logger: logging.Logger, logger_adapter: LoggerAdapter) -> Callable[[str], None]:
#     """Makes a function to log error messages caused by exceptions.
#
#     Iff LOCAL dev/test mode is enabled, then this returned function will log stacktraces.
#
#     Always logs the error message (:param:emsg) to the `error` level on the
#     :param:`logger_adapter`.
#     """
#     if LOCAL:
#
#         def _log_error(emsg: str) -> None:
#             logger.exception(emsg)
#             logger_adapter.error(emsg)
#
#     else:
#
#         def _log_error(emsg: str) -> None:
#             logger_adapter.error(emsg)
#
#     return _log_error
