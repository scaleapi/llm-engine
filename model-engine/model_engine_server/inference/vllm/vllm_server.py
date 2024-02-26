import argparse
import code
import json
import signal
import traceback
from typing import AsyncGenerator

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds
app = FastAPI()


@app.get("/healthz")
@app.get("/health")
def healthcheck():
    return "OK"


@app.post("/predict")
@app.post("/stream")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[str, None]:
        last_output_text = ""
        async for request_output in results_generator:
            ret = {
                "text": request_output.outputs[-1].text[len(last_output_text) :],
                "count_prompt_tokens": len(request_output.prompt_token_ids),
                "count_output_tokens": len(request_output.outputs[0].token_ids),
                "log_probs": (
                    request_output.outputs[0].logprobs[-1] if sampling_params.logprobs else None
                ),
                "finished": request_output.finished,
            }
            last_output_text = request_output.outputs[-1].text
            yield f"data:{json.dumps(ret)}\n\n"

    async def abort_request() -> None:
        await engine.abort(request_id)

    if stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)

    # Non-streaming case
    final_output = None
    tokens = []
    last_output_text = ""
    async for request_output in results_generator:
        tokens.append(request_output.outputs[-1].text[len(last_output_text) :])
        last_output_text = request_output.outputs[-1].text
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    ret = {
        "text": final_output.outputs[0].text,
        "count_prompt_tokens": len(final_output.prompt_token_ids),
        "count_output_tokens": len(final_output.outputs[0].token_ids),
        "log_probs": final_output.outputs[0].logprobs,
        "tokens": tokens,
    }
    return Response(content=json.dumps(ret))


def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d = {"_frame": frame}  # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)

    i = code.InteractiveConsole(d)
    message = "Signal received : entering python shell.\nTraceback:\n"
    message += "".join(traceback.format_stack(frame))
    i.interact(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)  # None == IPv4 / IPv6 dualstack
    parser.add_argument("--port", type=int, default=5005)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    signal.signal(signal.SIGUSR1, debug)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
