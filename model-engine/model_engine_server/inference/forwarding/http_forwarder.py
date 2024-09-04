import argparse
import json
import os
import subprocess
from functools import lru_cache, partial
from typing import Dict, Optional, Tuple

from fastapi import BackgroundTasks, Depends, FastAPI
from model_engine_server.common.concurrency_limiter import MultiprocessingConcurrencyLimiter
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.inference.forwarding.forwarding import (
    Forwarder,
    LoadForwarder,
    LoadStreamingForwarder,
    StreamingForwarder,
    load_named_config,
)
from sse_starlette.sse import EventSourceResponse

logger = make_logger(logger_name())
app = FastAPI()


@app.get("/healthz")
@app.get("/readyz")
def healthcheck():
    return "OK"


def get_config():
    overrides = os.getenv("CONFIG_OVERRIDES")
    config_overrides = None
    if overrides is not None:
        config_overrides = overrides.split(";")
    return load_named_config(
        os.getenv("CONFIG_FILE"),
        config_overrides,
    )


def get_forwarder_loader(destination_path: Optional[str] = None):
    config = get_config()["sync"]
    if destination_path:
        config["predict_route"] = destination_path
    forwarder_loader = LoadForwarder(**config)
    return forwarder_loader


def get_streaming_forwarder_loader(destination_path: Optional[str] = None):
    config = get_config()["stream"]
    if destination_path:
        config["predict_route"] = destination_path
    streaming_forwarder_loader = LoadStreamingForwarder(**config)
    return streaming_forwarder_loader


@lru_cache()
def get_concurrency_limiter():
    config = get_config()
    concurrency = int(config.get("max_concurrency", 100))
    return MultiprocessingConcurrencyLimiter(
        concurrency=concurrency, fail_on_concurrency_limit=True
    )


@lru_cache()
def load_forwarder(destination_path: Optional[str] = None):
    return get_forwarder_loader(destination_path).load(None, None)


@lru_cache()
def load_streaming_forwarder(destination_path: Optional[str] = None):
    return get_streaming_forwarder_loader(destination_path).load(None, None)


@app.post("/predict")
def predict(
    request: EndpointPredictV1Request,
    background_tasks: BackgroundTasks,
    forwarder=Depends(load_forwarder),
    limiter=Depends(get_concurrency_limiter),
):
    with limiter:
        try:
            response = forwarder(request.dict())
            background_tasks.add_task(
                forwarder.post_inference_hooks_handler.handle, request, response
            )
            return response
        except Exception:
            logger.error(f"Failed to decode payload from: {request}")
            raise


@app.post("/stream")
async def stream(
    request: EndpointPredictV1Request,
    forwarder=Depends(load_streaming_forwarder),
    limiter=Depends(get_concurrency_limiter),
):
    with limiter:
        try:
            payload = request.dict()
        except Exception:
            logger.error(f"Failed to decode payload from: {request}")
            raise
        else:
            logger.debug(f"Received request: {payload}")

        # has internal error logging for each processing stage
        responses = forwarder(payload)

        async def event_generator():
            for response in responses:
                yield {"data": json.dumps(response)}

        return EventSourceResponse(event_generator())


# This route is a catch-all for any requests that don't match the /predict or /stream routes
# It will treat the request as a streaming request if the "stream" body parameter is set to true
# NOTE: it is important for this to be defined AFTER the /predict and /stream endpoints
# because FastAPI will match the first route that matches the request path
async def predict_or_stream(
    request: EndpointPredictV1Request,
    background_tasks: BackgroundTasks,
    sync_forwarder: Optional[Forwarder],
    stream_forwarder: Optional[StreamingForwarder],
    limiter=Depends(get_concurrency_limiter),
):
    if stream_forwarder and request.args and request.args.root.get("stream", False):
        return stream(request, stream_forwarder, limiter)
    elif sync_forwarder:
        return predict(request, background_tasks, sync_forwarder, limiter)
    else:
        raise Exception("No forwarder configured for this route")


def add_extra_routes():
    """Read extra_routes from config and dynamically add routes to app"""
    config = get_config()
    # aggregate a list of routes -> (sync, stream) support
    extra_routes: Dict[str, Tuple[Optional[Forwarder], Optional[StreamingForwarder]]] = dict()
    for route in config.get("sync", {}).get("extra_routes", []):
        extra_routes[route] = (load_forwarder(route), None)
    for route in config.get("stream", {}).get("extra_routes", []):
        extra_routes[route] = (
            extra_routes.get(route, (None, None))[0],
            load_streaming_forwarder(route),
        )

    for route, (sync_forwarder, stream_forwarder) in extra_routes.items():
        fn = partial(
            predict_or_stream,
            sync_forwarder=sync_forwarder,
            stream_forwarder=stream_forwarder,
        )
        app.add_api_route(path=route, endpoint=fn, methods=["POST"])


def entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--host", type=str, default="[::]")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--set", type=str, action="append")
    parser.add_argument("--graceful-timeout", type=int, default=600)

    args, extra_args = parser.parse_known_args()

    values = [f"CONFIG_FILE={args.config}"]
    if args.set is not None:
        values.append(f"CONFIG_OVERRIDES={';'.join(args.set)}")
    envs = []
    for v in values:
        envs.extend(["--env", v])

    add_extra_routes()

    command = [
        "gunicorn",
        "--bind",
        f"{args.host}:{args.port}",
        "--timeout",
        "1200",
        "--keep-alive",
        "2",
        "--worker-class",
        "uvicorn.workers.UvicornWorker",
        "--workers",
        str(args.num_workers),
        "--graceful-timeout",
        str(args.graceful_timeout),
        *envs,
        "model_engine_server.inference.forwarding.http_forwarder:app",
        *extra_args,
    ]
    subprocess.run(command)


if __name__ == "__main__":
    entrypoint()
