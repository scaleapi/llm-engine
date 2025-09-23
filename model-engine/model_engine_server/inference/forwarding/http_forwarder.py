import argparse
import asyncio
import os
import signal
from functools import lru_cache
from typing import Any, Dict, Optional

import orjson
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from model_engine_server.common.concurrency_limiter import MultiprocessingConcurrencyLimiter
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.inference.forwarding.forwarding import (
    Forwarder,
    LoadForwarder,
    LoadPassthroughForwarder,
    LoadStreamingForwarder,
    PassthroughForwarder,
    StreamingForwarder,
    load_named_config,
)
from sse_starlette import EventSourceResponse

logger = make_logger(logger_name())


def get_config():
    overrides = os.getenv("CONFIG_OVERRIDES")
    config_overrides = None
    if overrides is not None:
        config_overrides = overrides.split(";")
    return load_named_config(
        os.getenv("CONFIG_FILE"),
        config_overrides,
    )


def get_forwarder_loader(destination_path: Optional[str] = None) -> LoadForwarder:
    config = get_config()["sync"]
    if "extra_routes" in config:
        del config["extra_routes"]
    if destination_path:
        config["predict_route"] = destination_path
    if "forwarder_type" in config:
        del config["forwarder_type"]
    forwarder_loader = LoadForwarder(**config)
    return forwarder_loader


def get_streaming_forwarder_loader(
    destination_path: Optional[str] = None,
) -> LoadStreamingForwarder:
    config = get_config()["stream"]
    if "extra_routes" in config:
        del config["extra_routes"]
    if destination_path:
        config["predict_route"] = destination_path
    if "forwarder_type" in config:
        del config["forwarder_type"]
    streaming_forwarder_loader = LoadStreamingForwarder(**config)
    return streaming_forwarder_loader


def get_stream_passthrough_forwarder_loader(
    destination_path: Optional[str] = None,
) -> LoadPassthroughForwarder:
    config = {}
    stream_config = get_config().get("stream", {})
    for key in ["user_port", "user_hostname", "healthcheck_route"]:
        config[key] = stream_config[key]
    if destination_path:
        config["passthrough_route"] = destination_path

    passthrough_forwarder_loader = LoadPassthroughForwarder(**config)
    return passthrough_forwarder_loader


def get_sync_passthrough_forwarder_loader(
    destination_path: Optional[str] = None,
) -> LoadPassthroughForwarder:
    config = {}
    sync_config = get_config().get("sync", {})
    for key in ["user_port", "user_hostname", "healthcheck_route"]:
        config[key] = sync_config[key]
    if destination_path:
        config["passthrough_route"] = destination_path

    passthrough_forwarder_loader = LoadPassthroughForwarder(**config)
    return passthrough_forwarder_loader


@lru_cache()
def get_concurrency_limiter() -> MultiprocessingConcurrencyLimiter:
    config = get_config()
    concurrency = int(config.get("max_concurrency", 100))
    return MultiprocessingConcurrencyLimiter(
        concurrency=concurrency, fail_on_concurrency_limit=True
    )


@lru_cache()
def load_forwarder(destination_path: Optional[str] = None) -> Forwarder:
    return get_forwarder_loader(destination_path).load(None, None)


@lru_cache()
def load_streaming_forwarder(destination_path: Optional[str] = None) -> StreamingForwarder:
    return get_streaming_forwarder_loader(destination_path).load(None, None)


@lru_cache()
def load_stream_passthrough_forwarder(
    destination_path: Optional[str] = None,
) -> PassthroughForwarder:
    return get_stream_passthrough_forwarder_loader(destination_path).load(None, None)


@lru_cache()
def load_sync_passthrough_forwarder(destination_path: Optional[str] = None) -> PassthroughForwarder:
    return get_sync_passthrough_forwarder_loader(destination_path).load(None, None)


HOP_BY_HOP_HEADERS: list[str] = [
    "proxy-authenticate",
    "proxy-authorization",
    "content-length",
    "content-encoding",
]


def sanitize_response_headers(headers: dict, force_cache_bust: bool = False) -> dict:
    lower_headers = {k.lower(): v for k, v in headers.items()}
    # Delete hop by hop headers that should not be forwarded
    for header in HOP_BY_HOP_HEADERS:
        if header in lower_headers:
            del lower_headers[header]

    if force_cache_bust:
        # force clients to refetch resources
        lower_headers["cache-control"] = "no-store"
        if "etag" in lower_headers:
            del lower_headers["etag"]
    return lower_headers


async def completions_endpoint(
    request: EndpointPredictV1Request,
    background_tasks: BackgroundTasks,
    sync_forwarder: Forwarder = Depends(load_forwarder),
    stream_forwarder: StreamingForwarder = Depends(load_streaming_forwarder),
    limiter: MultiprocessingConcurrencyLimiter = Depends(get_concurrency_limiter),
):
    """OpenAI-compatible completions endpoint that handles both sync and streaming requests."""
    with limiter:
        try:
            payload = request.model_dump()
        except Exception:
            logger.error(f"Failed to decode payload from: {request}")
            raise

        # Determine if this is a streaming request
        is_stream = payload.get("args", {}).get("stream", False) if hasattr(payload.get("args", {}), 'get') else payload.get("stream", False)

        if is_stream:
            # Handle streaming request
            logger.debug(f"Received streaming request: {payload}")

            responses = stream_forwarder.forward(payload)
            # We fetch the first response to check if upstream request was successful
            # If it was not, this will raise the corresponding HTTPException
            # If it was, we will proceed to the event generator
            initial_response = await responses.__anext__()

            async def event_generator():
                yield {"data": orjson.dumps(initial_response).decode("utf-8")}

                async for response in responses:
                    yield {"data": orjson.dumps(response).decode("utf-8")}

            return EventSourceResponse(event_generator())
        else:
            # Handle sync request
            response = await sync_forwarder.forward(payload)
            if sync_forwarder.post_inference_hooks_handler:
                background_tasks.add_task(
                    sync_forwarder.post_inference_hooks_handler.handle, request, response
                )
            return response


async def passthrough_stream(
    request: Request,
    forwarder: PassthroughForwarder = Depends(get_stream_passthrough_forwarder_loader),
    limiter: MultiprocessingConcurrencyLimiter = Depends(get_concurrency_limiter),
):
    with limiter:
        response = forwarder.forward_stream(request)
        headers, status = await anext(response)
        headers = sanitize_response_headers(headers)

        async def content_generator():
            async for chunk in response:
                yield chunk

        return StreamingResponse(content_generator(), headers=headers, status_code=status)


async def passthrough_sync(
    request: Request,
    forwarder: PassthroughForwarder = Depends(get_sync_passthrough_forwarder_loader),
    limiter: MultiprocessingConcurrencyLimiter = Depends(get_concurrency_limiter),
):
    with limiter:
        response = await forwarder.forward_sync(request)
        headers = sanitize_response_headers(response.headers)
        content = await response.read()
        return Response(content=content, status_code=response.status, headers=headers)


async def serve_http(app: FastAPI, **uvicorn_kwargs: Any):  # pragma: no cover
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ", ".join(methods))

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        logger.info("Gracefully stopping http server")
        return server.shutdown()


async def run_server(args, **uvicorn_kwargs) -> None:  # pragma: no cover
    app = await init_app()
    shutdown_task = await serve_http(
        app,
        host=args.host,
        port=args.port,
        **uvicorn_kwargs,
    )

    await shutdown_task


async def init_app():
    app = FastAPI()

    def healthcheck():
        return "OK"

    def add_extra_sync_or_stream_routes(app: FastAPI):
        """Read extra_routes from config and dynamically add routes to app"""
        config = get_config()
        sync_forwarders: Dict[str, Forwarder] = dict()
        stream_forwarders: Dict[str, StreamingForwarder] = dict()
        for route in config.get("sync", {}).get("extra_routes", []):
            sync_forwarders[route] = load_forwarder(route)
        for route in config.get("stream", {}).get("extra_routes", []):
            stream_forwarders[route] = load_streaming_forwarder(route)

        all_routes = set(list(sync_forwarders.keys()) + list(stream_forwarders.keys()))

        for route in all_routes:

            def get_sync_forwarder(route=route):
                return sync_forwarders.get(route)

            def get_stream_forwarder(route=route):
                return stream_forwarders.get(route)

            # This route handles requests for extra routes defined in configuration
            # It will treat the request as a streaming request if the "stream" parameter is set to true in request args
            async def predict_or_stream(
                request: EndpointPredictV1Request,
                background_tasks: BackgroundTasks,
                sync_forwarder: Forwarder = Depends(get_sync_forwarder),
                stream_forwarder: StreamingForwarder = Depends(get_stream_forwarder),
                limiter=Depends(get_concurrency_limiter),
            ):
                """Handles requests for extra routes, routing to sync or streaming based on args."""
                if not request.args:
                    raise Exception("Request has no args")

                is_stream = request.args.root.get("stream", False)

                if is_stream and stream_forwarder:
                    # Handle streaming request using consolidated logic
                    with limiter:
                        try:
                            payload = request.model_dump()
                        except Exception:
                            logger.error(f"Failed to decode payload from: {request}")
                            raise

                        logger.debug(f"Received streaming request: {payload}")

                        responses = stream_forwarder.forward(payload)
                        initial_response = await responses.__anext__()

                        async def event_generator():
                            yield {"data": orjson.dumps(initial_response).decode("utf-8")}
                            async for response in responses:
                                yield {"data": orjson.dumps(response).decode("utf-8")}

                        return EventSourceResponse(event_generator())

                elif not is_stream and sync_forwarder:
                    # Handle sync request using consolidated logic
                    with limiter:
                        try:
                            payload = request.model_dump()
                        except Exception:
                            logger.error(f"Failed to decode payload from: {request}")
                            raise

                        response = await sync_forwarder.forward(payload)
                        if sync_forwarder.post_inference_hooks_handler:
                            background_tasks.add_task(
                                sync_forwarder.post_inference_hooks_handler.handle, request, response
                            )
                        return response
                else:
                    raise Exception("No forwarder configured for this route")

            logger.info(f"Adding route {route}")
            app.add_api_route(
                path=route,
                endpoint=predict_or_stream,
                methods=["POST"],
            )

    def add_stream_passthrough_routes(app: FastAPI):
        config = get_config()

        passthrough_forwarders: Dict[str, PassthroughForwarder] = dict()
        for route in config.get("stream", {}).get("extra_routes", []):
            passthrough_forwarders[route] = load_stream_passthrough_forwarder(route)

        for route in passthrough_forwarders:

            def get_passthrough_forwarder(route=route):
                return passthrough_forwarders.get(route)

            async def passthrough_route(
                request: Request,
                passthrough_forwarder: PassthroughForwarder = Depends(get_passthrough_forwarder),
                limiter=Depends(get_concurrency_limiter),
            ):
                return await passthrough_stream(request, passthrough_forwarder, limiter)

            app.add_api_route(
                path=route,
                endpoint=passthrough_route,
                methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
            )

    def add_sync_passthrough_routes(app: FastAPI):
        config = get_config()

        passthrough_forwarders: Dict[str, PassthroughForwarder] = dict()
        for route in config.get("sync", {}).get("extra_routes", []):
            passthrough_forwarders[route] = load_sync_passthrough_forwarder(route)

        for route in passthrough_forwarders:

            def get_passthrough_forwarder(route=route):
                return passthrough_forwarders.get(route)

            async def passthrough_route(
                request: Request,
                passthrough_forwarder: PassthroughForwarder = Depends(get_passthrough_forwarder),
                limiter=Depends(get_concurrency_limiter),
            ):
                return await passthrough_sync(request, passthrough_forwarder, limiter)

            app.add_api_route(
                path=route,
                endpoint=passthrough_route,
                methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
            )

    def add_extra_routes(app: FastAPI):
        config = get_config()
        if config.get("stream", {}).get("forwarder_type") == "passthrough":
            add_stream_passthrough_routes(app)
        elif config.get("sync", {}).get("forwarder_type") == "passthrough":
            add_sync_passthrough_routes(app)
        else:
            add_extra_sync_or_stream_routes(app)

    app.add_api_route(path="/healthz", endpoint=healthcheck, methods=["GET"])
    app.add_api_route(path="/readyz", endpoint=healthcheck, methods=["GET"])
    # Legacy /predict and /stream endpoints removed - using /v1/completions
    app.add_api_route(path="/v1/completions", endpoint=completions_endpoint, methods=["POST"])

    add_extra_routes(app)
    return app


def entrypoint():  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--set", type=str, action="append")
    parser.add_argument("--graceful-timeout", type=int, default=600)

    args, extra_args = parser.parse_known_args()

    os.environ["CONFIG_FILE"] = args.config
    if args.set is not None:
        os.environ["CONFIG_OVERRIDES"] = ";".join(args.set)

    asyncio.run(
        run_server(
            args,
            timeout_keep_alive=2,
            timeout_graceful_shutdown=args.graceful_timeout,
            workers=args.num_workers,
            *extra_args,
        )
    )


if __name__ == "__main__":
    entrypoint()
