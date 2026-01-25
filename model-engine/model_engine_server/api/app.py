import os
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import pytz
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.openapi.docs import get_redoc_html
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from model_engine_server.api.batch_jobs_v1 import batch_job_router_v1
from model_engine_server.api.dependencies import get_or_create_aioredis_pool
from model_engine_server.api.docker_image_batch_job_bundles_v1 import (
    docker_image_batch_job_bundle_router_v1,
)
from model_engine_server.api.files_v1 import file_router_v1
from model_engine_server.api.llms_v1 import llm_router_v1
from model_engine_server.api.model_bundles_v1 import model_bundle_router_v1
from model_engine_server.api.model_bundles_v2 import model_bundle_router_v2
from model_engine_server.api.model_endpoints_docs_v1 import model_endpoints_docs_router_v1
from model_engine_server.api.model_endpoints_v1 import model_endpoint_router_v1
from model_engine_server.api.tasks_v1 import inference_task_router_v1
from model_engine_server.api.triggers_v1 import trigger_router_v1
from model_engine_server.api.v2 import llm_router_v2
from model_engine_server.common.concurrency_limiter import MultiprocessingConcurrencyLimiter
from model_engine_server.core.loggers import (
    LoggerTagKey,
    LoggerTagManager,
    logger_name,
    make_logger,
)
from model_engine_server.core.tracing import get_tracing_gateway
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

logger = make_logger(logger_name())

# Allows us to make the Uvicorn worker concurrency in model_engine_server/api/worker.py very high
MAX_CONCURRENCY = 500

concurrency_limiter = MultiprocessingConcurrencyLimiter(
    concurrency=MAX_CONCURRENCY, fail_on_concurrency_limit=True
)

healthcheck_routes = ["/healthcheck", "/healthz", "/readyz"]

tracing_gateway = get_tracing_gateway()


class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            LoggerTagManager.set(LoggerTagKey.REQUEST_ID, str(uuid.uuid4()))
            LoggerTagManager.set(LoggerTagKey.REQUEST_SIZE, request.headers.get("content-length"))
            if tracing_gateway:
                tracing_gateway.extract_tracing_headers(request, service="model_engine_server")
            # we intentionally exclude healthcheck routes from the concurrency limiter
            if request.url.path in healthcheck_routes:
                return await call_next(request)
            with concurrency_limiter:
                return await call_next(request)
        except HTTPException as e:
            timestamp = datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S %Z")
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.detail,
                    "timestamp": timestamp,
                },
            )
        except Exception as e:
            tb_str = traceback.format_exception(e)
            request_id = LoggerTagManager.get(LoggerTagKey.REQUEST_ID)
            timestamp = datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S %Z")
            structured_log = {
                "error": str(e),
                "request_id": str(request_id),
                "traceback": "".join(tb_str),
            }
            logger.error("Unhandled exception: %s", structured_log)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal error occurred. Our team has been notified.",
                    "timestamp": timestamp,
                    "request_id": request_id,
                },
            )


app = FastAPI(
    title="launch",
    version="1.0.0",
    redoc_url=None,
    middleware=[Middleware(CustomMiddleware)],
)

app.include_router(batch_job_router_v1)
app.include_router(inference_task_router_v1)
app.include_router(model_bundle_router_v1)
app.include_router(model_bundle_router_v2)
app.include_router(model_endpoint_router_v1)
app.include_router(model_endpoints_docs_router_v1)
app.include_router(docker_image_batch_job_bundle_router_v1)
app.include_router(llm_router_v1)
app.include_router(file_router_v1)
app.include_router(trigger_router_v1)
app.include_router(llm_router_v2)


# Pattern-based schema renames for discriminated unions that generate ugly auto-names.
# Uses (prefix, suffix) tuples to match auto-generated names regardless of union members.
# This is robust to adding/removing members from the discriminated unions.
OPENAPI_SCHEMA_RENAME_PATTERNS: list[tuple[str, str, str]] = [
    # CreateLLMModelEndpointV1Request - matches any union starting with CreateVLLM...
    (
        "RootModel_Annotated_Union_Annotated_CreateVLLM",
        "__Discriminator__",
        "CreateLLMModelEndpointV1Request",
    ),
    # UpdateLLMModelEndpointV1Request - matches any union starting with UpdateVLLM...
    (
        "RootModel_Annotated_Union_Annotated_UpdateVLLM",
        "__Discriminator__",
        "UpdateLLMModelEndpointV1Request",
    ),
]


def _rename_openapi_schemas(openapi_schema: dict) -> dict:
    """
    Post-process OpenAPI schema to rename auto-generated discriminated union names
    to clean, user-friendly names.

    Uses pattern matching (prefix + suffix) to be robust against changes to the
    union members (e.g., adding a new inference framework).
    """
    if "components" not in openapi_schema or "schemas" not in openapi_schema["components"]:
        return openapi_schema

    schemas = openapi_schema["components"]["schemas"]

    # Build mapping of old->new names based on pattern matches
    renames = {}
    for schema_name in list(schemas.keys()):
        for prefix, suffix, new_name in OPENAPI_SCHEMA_RENAME_PATTERNS:
            if schema_name.startswith(prefix) and schema_name.endswith(suffix):
                renames[schema_name] = new_name
                break

    # Perform the renames
    for old_name, new_name in renames.items():
        if old_name in schemas:
            schemas[new_name] = schemas.pop(old_name)

    # Update all $ref references throughout the schema
    def update_refs(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "$ref" and isinstance(value, str):
                    for old_name, new_name in renames.items():
                        if old_name in value:
                            obj[key] = value.replace(old_name, new_name)
                            break
                else:
                    update_refs(value)
        elif isinstance(obj, list):
            for item in obj:
                update_refs(item)

    update_refs(openapi_schema)
    return openapi_schema


def _convert_openapi_31_to_30(obj: dict | list) -> None:
    """
    Recursively convert OpenAPI 3.1 patterns to OpenAPI 3.0 style for generator compatibility.

    Transforms:
        - anyOf: [{...}, {"type": "null"}] -> {..., "nullable": true}
        - Removes "const" when "enum" is present (3.1 feature not supported in 3.0)
    """
    if isinstance(obj, dict):
        # Handle anyOf with null type
        if "anyOf" in obj:
            anyof = obj["anyOf"]
            null_items = [
                item for item in anyof if isinstance(item, dict) and item.get("type") == "null"
            ]
            non_null_items = [
                item
                for item in anyof
                if not (isinstance(item, dict) and item.get("type") == "null")
            ]

            if null_items:  # Has at least one null type
                if len(non_null_items) == 1:
                    # Single non-null item - convert to that item with nullable
                    new_obj = dict(non_null_items[0])
                    new_obj["nullable"] = True
                    del obj["anyOf"]
                    obj.update(new_obj)
                    _convert_openapi_31_to_30(obj)
                    return
                elif len(non_null_items) > 1:
                    # Multiple non-null items - keep anyOf but remove null and add nullable
                    obj["anyOf"] = non_null_items
                    obj["nullable"] = True
                    for item in non_null_items:
                        _convert_openapi_31_to_30(item)
                    return

        # Remove "const" when "enum" is present (const is 3.1 only)
        if "const" in obj and "enum" in obj:
            del obj["const"]

        for value in obj.values():
            _convert_openapi_31_to_30(value)
    elif isinstance(obj, list):
        for item in obj:
            _convert_openapi_31_to_30(item)


def get_openapi_schema(openapi_30_compatible: bool = False) -> dict:
    """
    Generate OpenAPI schema with optional 3.0 compatibility processing.

    Args:
        openapi_30_compatible: If True, convert OpenAPI 3.1 patterns to 3.0 style
                               for compatibility with older code generators.
    """
    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    openapi_schema = _rename_openapi_schemas(openapi_schema)

    if openapi_30_compatible:
        _convert_openapi_31_to_30(openapi_schema)
        openapi_schema["openapi"] = "3.0.3"

    return openapi_schema


def custom_openapi():
    """Custom OpenAPI schema generator that renames discriminated union schemas."""
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    openapi_schema = _rename_openapi_schemas(openapi_schema)
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]


# TODO: Remove this once we have a better way to serve internal docs
INTERNAL_DOCS_PATH = str(Path(__file__).parents[3] / "launch_internal/site")
if os.path.exists(INTERNAL_DOCS_PATH):
    app.mount(
        "/python-docs",
        StaticFiles(directory=INTERNAL_DOCS_PATH, html=True),
        name="python-docs",
    )
    app.mount(  # pragma: no cover
        "/static-docs",
        StaticFiles(directory=INTERNAL_DOCS_PATH),
        name="static-docs",
    )


@app.get("/api", include_in_schema=False)
async def redoc_html():  # pragma: no cover
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static-docs/redoc.standalone.js",
    )


@app.on_event("startup")
def load_redis():
    get_or_create_aioredis_pool()


def healthcheck() -> Response:
    """Returns 200 if the app is healthy."""
    return Response(status_code=200)


for endpoint in healthcheck_routes:
    app.get(endpoint)(healthcheck)
