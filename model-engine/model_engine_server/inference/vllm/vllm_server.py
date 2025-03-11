import asyncio
import code
import json
import os
import signal
import socket
import subprocess
import traceback
from logging import Logger
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import Response, StreamingResponse
from vllm.engine.async_llm_engine import (
    AsyncEngineDeadError,
    build_guided_decoding_logits_processor_async,
)
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import build_app, build_async_engine_client, init_app_state
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.outputs import CompletionOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import Logprob
from vllm.utils import FlexibleArgumentParser, random_uuid
from vllm.version import __version__ as VLLM_VERSION

logger = Logger("vllm_server")

engine_client: EngineClient

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds

router = APIRouter()


@router.post("/predict")
@router.post("/stream")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    # check health before accepting request and fail fast if engine isn't healthy
    try:
        await engine_client.check_health()

        request_dict = await request.json()
        prompt = request_dict.pop("prompt")
        stream = request_dict.pop("stream", False)

        guided_decoding_backend = (
            await engine_client.get_decoding_config()
        ).guided_decoding_backend

        sampling_params = await build_guided_decoding_logits_processor_async(
            sampling_params=SamplingParams(**request_dict),
            tokenizer=await engine_client.get_tokenizer(lora_request=None),
            default_guided_backend=guided_decoding_backend,
            model_config=await engine_client.get_model_config(),
        )

        request_id = random_uuid()

        results_generator = engine_client.generate(prompt, sampling_params, request_id)

        async def abort_request() -> None:
            await engine_client.abort(request_id)

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
                        "log_probs": (
                            log_probs[-1] if log_probs and sampling_params.logprobs else None
                        ),
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
                await engine_client.abort(request_id)
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

    except AsyncEngineDeadError as e:
        logger.error(f"The vllm engine is dead, exiting the pod: {e}")
        os.kill(os.getpid(), signal.SIGINT)
        raise e


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
        logger.warn(f"Error getting GPU memory: {e}")
        return None


def check_unknown_startup_memory_usage():
    """Check for unknown memory usage at startup."""
    gpu_free_memory = get_gpu_free_memory()
    if gpu_free_memory is not None:
        min_mem = min(gpu_free_memory)
        max_mem = max(gpu_free_memory)
        if max_mem - min_mem > 10:
            logger.warn(
                f"WARNING: Unbalanced GPU memory usage at start up. This may cause OOM. Memory usage per GPU in MB: {gpu_free_memory}."
            )
            try:
                # nosemgrep
                output = subprocess.run(
                    ["fuser -v /dev/nvidia*"],
                    shell=False,
                    capture_output=True,
                    text=True,
                ).stdout
                logger.info(f"Processes using GPU: {output}")
            except Exception as e:
                logger.error(f"Error getting processes using GPU: {e}")


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


def format_logprobs(
    request_output: CompletionOutput,
) -> Optional[List[Dict[int, float]]]:
    """Given a request output, format the logprobs if they exist."""
    output_logprobs = request_output.outputs[0].logprobs
    if output_logprobs is None:
        return None

    def extract_logprobs(logprobs: Dict[int, Logprob]) -> Dict[int, float]:
        return {k: v.logprob for k, v in logprobs.items()}

    return [extract_logprobs(logprobs) for logprobs in output_logprobs]


def parse_args(parser: FlexibleArgumentParser):
    parser = make_arg_parser(parser)
    parser.add_argument("--attention-backend", type=str, help="The attention backend to use")
    return parser.parse_args()


async def run_server(args, **uvicorn_kwargs) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # nosemgrep
    temp_socket.bind(("", args.port))

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    global engine_client
    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)

        model_config = await engine_client.get_model_config()
        init_app_state(engine_client, model_config, app.state, args)

        temp_socket.close()
        app.include_router(router)

        shutdown_task = await serve_http(
            app,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task


if __name__ == "__main__":
    check_unknown_startup_memory_usage()

    parser = FlexibleArgumentParser()
    args = parse_args(parser)
    if args.attention_backend is not None:
        os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend
    asyncio.run(run_server(args))
