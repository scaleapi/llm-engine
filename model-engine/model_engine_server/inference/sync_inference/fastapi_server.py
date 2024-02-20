import traceback
from functools import wraps

from fastapi import FastAPI, HTTPException, Response, status
from model_engine_server.common.concurrency_limiter import MultiprocessingConcurrencyLimiter
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.inference.common import load_predict_fn_or_cls, run_predict
from model_engine_server.inference.sync_inference.constants import (
    CONCURRENCY,
    FAIL_ON_CONCURRENCY_LIMIT,
    NAME,
)

logger = make_logger(logger_name())


def with_concurrency_limit(concurrency_limiter: MultiprocessingConcurrencyLimiter):
    def _inner(flask_func):
        @wraps(flask_func)
        def _inner_2(*args, **kwargs):
            with concurrency_limiter:
                return flask_func(*args, **kwargs)

        return _inner_2

    return _inner


app = FastAPI(title=NAME)
concurrency_limiter = MultiprocessingConcurrencyLimiter(CONCURRENCY, FAIL_ON_CONCURRENCY_LIMIT)

# How does this interact with threads?
# Analogous to init_worker() inside async_inference
predict_fn = load_predict_fn_or_cls()


@app.get("/healthcheck")
@app.get("/healthz")
@app.get("/readyz")
def healthcheck():
    return Response(status_code=status.HTTP_200_OK)


@app.post("/predict")
@with_concurrency_limit(concurrency_limiter)
def predict(payload: EndpointPredictV1Request):
    """
    Assumption: payload is a JSON with format {"url": <url>, "args": <dictionary of args>, "returned_pickled": boolean}
    Returns: Results of running the predict function on the request url. See `run_predict`.
    """
    try:
        result = run_predict(predict_fn, payload)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail=dict(traceback=str(traceback.format_exc())))
