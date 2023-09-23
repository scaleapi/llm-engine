import os
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import pytz
from fastapi import FastAPI, Request, Response
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
from model_engine_server.core.loggers import (
    filename_wo_ext,
    get_request_id,
    make_logger,
    set_request_id,
)

logger = make_logger(filename_wo_ext(__name__))

app = FastAPI(title="launch", version="1.0.0", redoc_url="/api")

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


@app.middleware("http")
async def dispatch(request: Request, call_next):
    try:
        set_request_id(str(uuid.uuid4()))
        return await call_next(request)
    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        request_id = get_request_id()
        timestamp = datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S %Z")
        structured_log = {
            "error": str(e),
            "request_id": str(request_id),
            "traceback": "".join(tb_str),
        }
        logger.error("Unhandled exception: %s", structured_log)
        return JSONResponse(
            {
                "status_code": 500,
                "content": {
                    "error": "Internal error occurred. Our team has been notified.",
                    "timestamp": timestamp,
                    "request_id": request_id,
                },
            }
        )


# TODO: Remove this once we have a better way to serve internal docs
INTERNAL_DOCS_PATH = str(Path(__file__).parents[3] / "launch_internal/site")
if os.path.exists(INTERNAL_DOCS_PATH):
    app.mount(
        "/python-docs",
        StaticFiles(directory=INTERNAL_DOCS_PATH, html=True),
        name="python-docs",
    )


@app.on_event("startup")
def load_redis():
    get_or_create_aioredis_pool()


@app.get("/healthcheck")
@app.get("/healthz")
@app.get("/readyz")
def healthcheck() -> Response:
    """Returns 200 if the app is healthy."""
    return Response(status_code=200)
