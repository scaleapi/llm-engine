import argparse
import code
import json
import os
import signal
import subprocess
import traceback
from typing import AsyncGenerator, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import CompletionRequest as OpenAICompletionRequest
from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor
from vllm.outputs import CompletionOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import Logprob
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
    # check health before accepting request and fail fast if engine isn't healthy
    try:
        await engine.check_health()
    except Exception as e:
        print(f"The vllm engine is dead, exiting the pod: {e}")
        os.kill(os.getpid(), signal.SIGINT)

    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    guided_json = request_dict.pop("guided_json", None)
    guided_regex = request_dict.pop("guided_regex", None)
    guided_choice = request_dict.pop("guided_choice", None)
    guided_grammar = request_dict.pop("guided_grammar", None)
    sampling_params = SamplingParams(**request_dict)

    # Dummy request to get guided decode logit processor
    try:
        partial_openai_request = OpenAICompletionRequest.model_validate(
            {
                "model": "",
                "prompt": "",
                "guided_json": guided_json,
                "guided_regex": guided_regex,
                "guided_choice": guided_choice,
                "guided_grammar": guided_grammar,
            }
        )
    except Exception:
        raise HTTPException(
            status_code=400, detail="Bad request: failed to parse guided decoding parameters."
        )

    guided_decoding_backend = engine.engine.decoding_config.guided_decoding_backend
    guided_decode_logit_processor = await get_guided_decoding_logits_processor(
        guided_decoding_backend, partial_openai_request, await engine.get_tokenizer()
    )
    if guided_decode_logit_processor is not None:
        if sampling_params.logits_processors is None:
            sampling_params.logits_processors = []
        sampling_params.logits_processors.append(guided_decode_logit_processor)

    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    async def abort_request() -> None:
        await engine.abort(request_id)

    if stream:
        # Streaming case
        async def stream_results() -> AsyncGenerator[str, None]:
            last_output_text = ""
            async for request_output in results_generator:
                log_probs = format_logprobs(request_output)
                ret = {
                    "text": request_output.outputs[-1].text[len(last_output_text) :],
                    "count_prompt_tokens": len(request_output.prompt_token_ids),
                    "count_output_tokens": len(request_output.outputs[0].token_ids),
                    "log_probs": log_probs[-1] if log_probs and sampling_params.logprobs else None,
                    "finished": request_output.finished,
                }
                last_output_text = request_output.outputs[-1].text
                yield f"data:{json.dumps(ret)}\n\n"

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
        "log_probs": format_logprobs(final_output),
        "tokens": tokens,
    }
    return Response(content=json.dumps(ret))


def get_gpu_free_memory():
    """Get GPU free memory using nvidia-smi."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout
        gpu_memory = [int(x) for x in output.strip().split("\n")]
        return gpu_memory
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return None


def check_unknown_startup_memory_usage():
    """Check for unknown memory usage at startup."""
    gpu_free_memory = get_gpu_free_memory()
    if gpu_free_memory is not None:
        min_mem = min(gpu_free_memory)
        max_mem = max(gpu_free_memory)
        if max_mem - min_mem > 10:
            print(
                f"WARNING: Unbalanced GPU memory usage at start up. This may cause OOM. Memory usage per GPU in MB: {gpu_free_memory}."
            )
            try:
                # nosemgrep
                output = subprocess.run(
                    ["fuser -v /dev/nvidia*"], shell=True, capture_output=True, text=True
                ).stdout
                print(f"Processes using GPU: {output}")
            except Exception as e:
                print(f"Error getting processes using GPU: {e}")


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


def format_logprobs(request_output: CompletionOutput) -> Optional[List[Dict[int, float]]]:
    """Given a request output, format the logprobs if they exist."""
    output_logprobs = request_output.outputs[0].logprobs
    if output_logprobs is None:
        return None

    def extract_logprobs(logprobs: Dict[int, Logprob]) -> Dict[int, float]:
        return {k: v.logprob for k, v in logprobs.items()}

    return [extract_logprobs(logprobs) for logprobs in output_logprobs]


if __name__ == "__main__":
    check_unknown_startup_memory_usage()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)  # None == IPv4 / IPv6 dualstack
    parser.add_argument("--port", type=int, default=5005)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine.check_health()

    signal.signal(signal.SIGUSR1, debug)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
