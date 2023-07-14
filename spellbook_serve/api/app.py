from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles

from spellbook_serve.api.batch_jobs_v1 import batch_job_router_v1
from spellbook_serve.api.dependencies import get_or_create_aioredis_pool
from spellbook_serve.api.docker_image_batch_job_bundles_v1 import (
    docker_image_batch_job_bundle_router_v1,
)
from spellbook_serve.api.llms_v1 import llm_router_v1
from spellbook_serve.api.model_bundles_v1 import model_bundle_router_v1
from spellbook_serve.api.model_bundles_v2 import model_bundle_router_v2
from spellbook_serve.api.model_endpoints_docs_v1 import model_endpoints_docs_router_v1
from spellbook_serve.api.model_endpoints_v1 import model_endpoint_router_v1
from spellbook_serve.api.tasks_v1 import inference_task_router_v1

app = FastAPI(title="spellbook_serve", version="1.0.0", redoc_url="/api")

app.include_router(batch_job_router_v1)
app.include_router(inference_task_router_v1)
app.include_router(model_bundle_router_v1)
app.include_router(model_bundle_router_v2)
app.include_router(model_endpoint_router_v1)
app.include_router(model_endpoints_docs_router_v1)
app.include_router(docker_image_batch_job_bundle_router_v1)
app.include_router(llm_router_v1)


@app.on_event("startup")
def load_redis():
    get_or_create_aioredis_pool()


@app.get("/healthcheck")
@app.get("/healthz")
@app.get("/readyz")
def healthcheck() -> Response:
    """Returns 200 if the app is healthy."""
    return Response(status_code=200)
