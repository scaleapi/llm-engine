# Capture Python start time BEFORE heavy imports (for python_init metric)
import os
import time

_PYTHON_START_TIME = time.perf_counter()
_PYTHON_START_TIME_NS = time.time_ns()

# Startup metrics feature gate (check early to avoid unnecessary imports)
ENABLE_STARTUP_METRICS = os.environ.get("ENABLE_STARTUP_METRICS", "").lower() == "true"

# Now do heavy imports (noqa: E402 - intentional late import for startup time measurement)
import asyncio  # noqa: E402
import code  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402
from logging import Logger  # noqa: E402

from vllm.engine.protocol import EngineClient  # noqa: E402
from vllm.entrypoints.openai.api_server import run_server  # noqa: E402
from vllm.entrypoints.openai.cli_args import make_arg_parser  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402

logger = Logger("vllm_server")

engine_client: EngineClient

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds


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


def _health_check_loop(metrics, vllm_init_start_ns: int, host: str = "localhost", port: int = 5005):
    """
    Background thread that polls /health endpoint to detect when server is ready.
    Once ready, records the vllm_init span and startup complete metric.
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
                    # Server is ready - record vllm_init span
                    end_time_ns = time.time_ns()
                    metrics.record_vllm_init(vllm_init_start_ns, end_time_ns)

                    # Record startup complete (emits in_container_startup span)
                    total_duration = metrics.complete()
                    print(f"[STARTUP METRICS] Server ready after {total_duration:.2f}s total")
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionRefusedError, OSError):
            pass

        attempt += 1
        time.sleep(1)

    print("[STARTUP METRICS] WARNING: Health check timed out, server may not be ready")


async def _run_instrumented_server(args):
    """Run vLLM server with startup instrumentation.

    Emits spans for:
    - python_init: From module start to vllm_init start
    - vllm_init: From vllm_init start to server ready (health check passes)

    Download span is emitted by vllm_startup_wrapper.py before exec'ing into this.
    """
    from startup_telemetry import VLLMStartupMetrics

    # Initialize metrics (reads CONTAINER_START_TS from env)
    metrics = VLLMStartupMetrics.init()

    if metrics.enabled:
        print(f"[STARTUP METRICS] trace_id={metrics.trace_id}")

    # Record Python init time (from module start to now)
    python_init_duration = time.perf_counter() - _PYTHON_START_TIME
    metrics.record_python_init(python_init_duration)
    print(f"[STARTUP METRICS] Python init: {python_init_duration:.2f}s")

    # Mark vllm_init start time
    vllm_init_start_ns = time.time_ns()

    # Start health check thread to detect when server is ready
    # This will record vllm_init span and call metrics.complete() when /health returns 200
    port = getattr(args, "port", 5005)
    health_thread = threading.Thread(
        target=_health_check_loop,
        args=(metrics, vllm_init_start_ns, "localhost", port),
        daemon=True,
    )
    health_thread.start()

    # Start vLLM server (blocking - runs until shutdown)
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
