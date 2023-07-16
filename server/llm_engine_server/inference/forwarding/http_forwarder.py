import argparse
import json
import os
import subprocess
from functools import lru_cache
from typing import Any, List

import yaml
from fastapi import Depends, FastAPI
from llm_engine_server.common.dtos.tasks import EndpointPredictV1Request
from llm_engine_server.core.loggers import logger_name, make_logger
from llm_engine_server.inference.forwarding.forwarding import LoadForwarder, LoadStreamingForwarder
from sse_starlette.sse import EventSourceResponse

logger = make_logger(logger_name())
app = FastAPI()


def _set_value(config: dict, key_path: List[str], value: Any) -> None:
    """
    Modifies config by setting the value at config[key_path[0]][key_path[1]]... to be `value`.
    """
    key = key_path[0]
    if len(key_path) == 1:
        config[key] = value
    else:
        if key not in config:
            config[key] = dict()
        _set_value(config[key], key_path[1:], value)


def _substitute_config_overrides(config: dict, config_overrides: List[str]) -> None:
    """
    Modifies config based on config_overrides.

    config_overrides should be a list of strings of the form `key=value`,
    where `key` can be of the form `key1.key2` to denote a substitution for config[key1][key2]
    (nesting can be arbitrarily deep).
    """
    for override in config_overrides:
        split = override.split("=")
        if len(split) != 2:
            raise ValueError(f"Config override {override} must contain exactly one =")
        key_path, value = split
        try:
            _set_value(config, key_path.split("."), value)
        except Exception as e:
            raise ValueError(f"Error setting {key_path} to {value} in {config}") from e


def _load_named_config(config_uri, config_overrides=None):
    with open(config_uri, "rt") as rt:
        if config_uri.endswith(".json"):
            return json.load(rt)
        else:
            c = yaml.safe_load(rt)
            if config_overrides:
                _substitute_config_overrides(c, config_overrides)
            if len(c) == 1:
                name = list(c.keys())[0]
                c = c[name]
                if "name" not in c:
                    c["name"] = name
            return c


@app.get("/healthz")
@app.get("/readyz")
def healthcheck():
    return "OK"


def get_config():
    overrides = os.getenv("CONFIG_OVERRIDES")
    config_overrides = None
    if overrides is not None:
        config_overrides = overrides.split(";")
    return _load_named_config(
        os.getenv("CONFIG_FILE"),
        config_overrides,
    )


def get_forwarder_loader():
    config = get_config()
    forwarder_loader = LoadForwarder(**config["sync"])
    return forwarder_loader


def get_streaming_forwarder_loader():
    config = get_config()
    streaming_forwarder_loader = LoadStreamingForwarder(**config["stream"])
    return streaming_forwarder_loader


@lru_cache()
def load_forwarder():
    return get_forwarder_loader().load(None, None)


@lru_cache()
def load_streaming_forwarder():
    return get_streaming_forwarder_loader().load(None, None)


@app.post("/predict")
def predict(request: EndpointPredictV1Request, forwarder=Depends(load_forwarder)):
    return forwarder(request.dict())


@app.post("/stream")
async def stream(request: EndpointPredictV1Request, forwarder=Depends(load_streaming_forwarder)):
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


def entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--host", type=str, default="[::]")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--set", type=str, action="append")

    args = parser.parse_args()

    values = [f"CONFIG_FILE={args.config}"]
    if args.set is not None:
        values.append(f"CONFIG_OVERRIDES={';'.join(args.set)}")
    envs = []
    for v in values:
        envs.extend(["--env", v])

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
        *envs,
        "llm_engine_server.inference.forwarding.http_forwarder:app",
    ]
    subprocess.run(command)


if __name__ == "__main__":
    entrypoint()
