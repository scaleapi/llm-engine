import traceback
from functools import wraps

from fastapi import BackgroundTasks, FastAPI, HTTPException, Response, status
from model_engine_server.common.concurrency_limiter import MultiprocessingConcurrencyLimiter
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.inference.common import (
    get_endpoint_config,
    load_predict_fn_or_cls,
    run_predict,
)
from model_engine_server.inference.infra.gateways.datadog_inference_monitoring_metrics_gateway import (
    DatadogInferenceMonitoringMetricsGateway,
)
from model_engine_server.inference.post_inference_hooks import PostInferenceHooksHandler
from model_engine_server.inference.sync_inference.constants import (
    CONCURRENCY,
    FAIL_ON_CONCURRENCY_LIMIT,
    NAME,
)

logger = make_logger(logger_name())

# do we still need this file
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
endpoint_config = get_endpoint_config()
hooks = PostInferenceHooksHandler(
    endpoint_name=endpoint_config.endpoint_name,
    bundle_name=endpoint_config.bundle_name,
    post_inference_hooks=endpoint_config.post_inference_hooks,
    user_id=endpoint_config.user_id,
    billing_queue=endpoint_config.billing_queue,
    billing_tags=endpoint_config.billing_tags,
    default_callback_url=endpoint_config.default_callback_url,
    default_callback_auth=endpoint_config.default_callback_auth,
    monitoring_metrics_gateway=DatadogInferenceMonitoringMetricsGateway(),
)


@app.get("/healthcheck")
@app.get("/healthz")
@app.get("/readyz")
def healthcheck():
    return Response(status_code=status.HTTP_200_OK)


@app.post("/predict")
@with_concurrency_limit(concurrency_limiter)
def predict(payload: EndpointPredictV1Request, background_tasks: BackgroundTasks):
    """
    Assumption: payload is a JSON with format {"url": <url>, "args": <dictionary of args>, "returned_pickled": boolean}
    Returns: Results of running the predict function on the request url. See `run_predict`.
    """
    try:
        result = run_predict(predict_fn, payload)
        background_tasks.add_task(hooks.handle, payload, result)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail=dict(traceback=str(traceback.format_exc())))
