"""
This file is for testing purposes only. It serves as simple server to mock a deployed model. 
"""
import argparse
import subprocess

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

app = FastAPI()


@app.get("/healthz")
@app.get("/readyz")
def healthcheck():
    return "OK"


@app.post("/predict")
async def predict(request: Request):
    return await request.json()


@app.post("/predict500")
async def predict500(request: Request):
    response = JSONResponse(content=await request.json(), status_code=500)
    return response


@app.post("/stream")
async def stream(request: Request):
    value = (await request.body()).decode()
    return EventSourceResponse([{"data": value}].__iter__())


def entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--host", type=str, default="[::]")
    parser.add_argument("--port", type=int, default=5009)

    args, extra_args = parser.parse_known_args()

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
        "model_engine_server.inference.forwarding.echo_server:app",
        *extra_args,
    ]
    subprocess.run(command)


if __name__ == "__main__":
    entrypoint()
