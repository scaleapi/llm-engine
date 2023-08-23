import os
from functools import wraps
from threading import BoundedSemaphore
from typing import Optional

import waitress
from flask import Flask, Response, abort, request
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from model_engine_server.core.loggers import filename_wo_ext, make_logger
from model_engine_server.inference.common import load_predict_fn_or_cls, run_predict

logger = make_logger(filename_wo_ext(__file__))

NAME = "hosted-inference-sync-service"
CONCURRENCY = 2  # TODO read from env var?? what's our api
NUM_THREADS = CONCURRENCY + 1  # Extra thread for rejecting above-concurrency requests
FAIL_ON_CONCURRENCY_LIMIT = True  # TODO read from env var??
PORT = os.environ["PORT"]


class FlaskConcurrencyLimiter:
    def __init__(self, concurrency: Optional[int], fail_on_concurrency_limit: bool):
        if concurrency is not None:
            if concurrency < 1:
                raise ValueError("Concurrency should be at least 1")
            self.semaphore: Optional[BoundedSemaphore] = BoundedSemaphore(value=concurrency)
            self.blocking = (
                not fail_on_concurrency_limit
            )  # we want to block if we want to queue up requests
        else:
            self.semaphore = None
            self.blocking = False  # Unused

    def __enter__(self):
        logger.debug("Entering concurrency limiter semaphore")
        if self.semaphore and not self.semaphore.acquire(blocking=self.blocking):
            logger.warning("Too many requests, returning 429")
            abort(429)
            # Just raises an HTTPException.
            # __exit__ should not run; otherwise the release() doesn't have an acquire()

    def __exit__(self, type, value, traceback):
        logger.debug("Exiting concurrency limiter semaphore")
        if self.semaphore:
            self.semaphore.release()


def with_concurrency_limit(concurrency_limiter: FlaskConcurrencyLimiter):
    def _inner(flask_func):
        @wraps(flask_func)
        def _inner_2(*args, **kwargs):
            with concurrency_limiter:
                return flask_func(*args, **kwargs)

        return _inner_2

    return _inner


app = Flask(NAME)
concurrency_limiter = FlaskConcurrencyLimiter(CONCURRENCY, FAIL_ON_CONCURRENCY_LIMIT)

# How does this interact with threads?
# Analogous to init_worker() inside async_inference
predict_fn = load_predict_fn_or_cls()


@app.route("/healthcheck", methods=["GET"])
@app.route("/healthz", methods=["GET"])
@app.route("/readyz", methods=["GET"])
def healthcheck():
    return Response(status=200, headers={})


@app.route("/predict", methods=["POST"])
@with_concurrency_limit(concurrency_limiter)
def predict():
    """
    Assumption: payload is a JSON with format {"url": <url>, "args": <dictionary of args>, "returned_pickled": boolean}
    Returns: Results of running the predict function on the request url. See `run_predict`.

    """
    try:
        payload = request.get_json()
        payload_pydantic = EndpointPredictV1Request.parse_obj(payload)
    except Exception:
        logger.error(f"Failed to decode payload from: {request}")
        raise
    else:
        logger.debug(f"Received request: {payload}")

    return run_predict(predict_fn, payload_pydantic)


if __name__ == "__main__":
    waitress.serve(app, port=PORT, url_scheme="https", threads=NUM_THREADS)
