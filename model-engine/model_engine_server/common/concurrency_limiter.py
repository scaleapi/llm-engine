from multiprocessing import BoundedSemaphore
from multiprocessing.synchronize import BoundedSemaphore as BoundedSemaphoreType
from typing import Optional

from fastapi import HTTPException
from model_engine_server.core.loggers import logger_name, make_logger

logger = make_logger(logger_name())


class MultiprocessingConcurrencyLimiter:
    def __init__(self, concurrency: Optional[int], fail_on_concurrency_limit: bool):
        if concurrency is not None:
            if concurrency < 1:
                raise ValueError("Concurrency should be at least 1")
            self.semaphore: Optional[BoundedSemaphoreType] = BoundedSemaphore(value=concurrency)
            self.blocking = (
                not fail_on_concurrency_limit
            )  # we want to block if we want to queue up requests
        else:
            self.semaphore = None
            self.blocking = False  # Unused

    def __enter__(self):
        logger.debug("Entering concurrency limiter semaphore")
        if self.semaphore and not self.semaphore.acquire(block=self.blocking):
            logger.warning("Too many requests, returning 429")
            raise HTTPException(status_code=429, detail="Too many requests")
            # Just raises an HTTPException.
            # __exit__ should not run; otherwise the release() doesn't have an acquire()

    def __exit__(self, type, value, traceback):
        logger.debug("Exiting concurrency limiter semaphore")
        if self.semaphore:
            self.semaphore.release()
