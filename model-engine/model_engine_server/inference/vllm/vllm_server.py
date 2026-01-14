# Capture Python start time BEFORE heavy imports (for python_init metric)
import os
import time

_PYTHON_START_TIME = time.perf_counter()
_PYTHON_START_TIME_NS = time.time_ns()

# Startup metrics feature gate (check early to avoid unnecessary imports)
ENABLE_STARTUP_METRICS = os.environ.get("ENABLE_STARTUP_METRICS", "").lower() == "true"

# Now do heavy imports
import asyncio
import code
import subprocess
import threading
import traceback
from logging import Logger

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = Logger("vllm_server")

engine_client: EngineClient

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds

# Global to track vLLM init timing (only used when metrics enabled)
_VLLM_INIT_START_TIME = None
_VLLM_INIT_START_TIME_NS = None

# Legacy endpoints /predit and /stream removed - using vLLM's native OpenAI-compatible endpoints instead
# All requests now go through /v1/completions, /v1/chat/completions, etc.


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


def parse_args(parser: FlexibleArgumentParser):
    parser = make_arg_parser(parser)
    parser.add_argument("--attention-backend", type=str, help="The attention backend to use")
    return parser.parse_args()


def get_gpu_type():
    """Get GPU type from nvidia-smi."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        ).stdout
        return output.strip().split("\n")[0]
    except Exception as e:
        logger.warning(f"Error getting GPU type: {e}")
        return "unknown"


def _health_check_loop(telemetry, host: str = "localhost", port: int = 5005):
    """
    Background thread that polls /health endpoint to detect when server is ready.
    Once ready, records the startup complete metric and vllm_init span.
    """
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/health"
    max_attempts = 600  # 10 minutes max
    attempt = 0

    while attempt < max_attempts:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    # Server is ready!
                    ready_time = time.perf_counter()
                    ready_time_ns = time.time_ns()

                    # Record vllm_init span
                    if _VLLM_INIT_START_TIME_NS:
                        vllm_init_duration = ready_time - _VLLM_INIT_START_TIME
                        telemetry.create_child_span(
                            "vllm_init",
                            start_time_ns=_VLLM_INIT_START_TIME_NS,
                            end_time_ns=ready_time_ns,
                            attributes={"vllm_init_duration_seconds": vllm_init_duration},
                        )
                        telemetry.record_metric("vllm_init_duration", vllm_init_duration)
                        print(f"[STARTUP METRICS] vLLM init took {vllm_init_duration:.2f}s")

                    # Record total startup (also closes root span)
                    duration = telemetry.record_startup_complete()
                    print(f"[STARTUP METRICS] Server ready after {duration:.2f}s total")
                    telemetry.flush()
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionRefusedError, OSError):
            pass

        attempt += 1
        time.sleep(1)

    print("[STARTUP METRICS] WARNING: Health check timed out, server may not be ready")


async def _run_instrumented_server(args):
    """Run vLLM server with startup instrumentation.

    Emits two spans:
    - python_init: From module start (_PYTHON_START_TIME) to vllm_init start
    - vllm_init: From vllm_init start to server ready (health check passes)

    Download span is emitted by vllm_startup_wrapper.py before exec'ing into this.
    """
    global _VLLM_INIT_START_TIME, _VLLM_INIT_START_TIME_NS

    # Import telemetry modules (only when metrics enabled)
    from startup_telemetry import StartupContext, init_startup_telemetry

    # Initialize telemetry
    ctx = StartupContext(
        endpoint_name=os.environ.get("ENDPOINT_NAME", "unknown"),
        model_name=os.environ.get("MODEL_NAME", getattr(args, "model", "unknown")),
        gpu_type=os.environ.get("GPU_TYPE", get_gpu_type()),
        num_gpus=int(os.environ.get("NUM_GPUS", "1")),
        region=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")),
        pod_uid=os.environ.get("POD_UID", "unknown"),
        pod_name=os.environ.get("POD_NAME", "unknown"),
        node_name=os.environ.get("NODE_NAME", "unknown"),
    )

    telemetry = init_startup_telemetry(ctx)

    # Mark when vLLM init starts (after imports, before run_server)
    _VLLM_INIT_START_TIME = time.perf_counter()
    _VLLM_INIT_START_TIME_NS = time.time_ns()

    # Emit python_init span (from module start to now)
    python_init_duration = _VLLM_INIT_START_TIME - _PYTHON_START_TIME
    telemetry.create_child_span(
        "python_init",
        start_time_ns=_PYTHON_START_TIME_NS,
        end_time_ns=_VLLM_INIT_START_TIME_NS,
        attributes={
            "description": "Python startup, module imports, arg parsing",
            "duration_seconds": python_init_duration,
        },
    )
    telemetry.record_metric("python_init_duration", python_init_duration)
    print(f"[STARTUP METRICS] Python init: {python_init_duration:.2f}s")

    # Start health check thread to detect when server is ready
    port = getattr(args, "port", 5005)
    health_thread = threading.Thread(
        target=_health_check_loop, args=(telemetry, "localhost", port), daemon=True
    )
    health_thread.start()

    # Start vLLM server (blocking)
    await run_server(args)


if __name__ == "__main__":
    check_unknown_startup_memory_usage()

    parser = FlexibleArgumentParser()
    args = parse_args(parser)
    if args.attention_backend is not None:
        os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend

    if ENABLE_STARTUP_METRICS:
        print("[STARTUP METRICS] Startup metrics enabled")
        asyncio.run(_run_instrumented_server(args))
    else:
        # Standard server without instrumentation
        asyncio.run(run_server(args))
