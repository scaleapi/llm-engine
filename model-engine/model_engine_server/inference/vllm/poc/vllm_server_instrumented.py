"""
Instrumented vLLM server with startup metrics.

This wraps the standard vLLM server to capture startup timing metrics
and emit them via OpenTelemetry to Datadog.
"""

import asyncio
import os
import subprocess
import threading
import time
from logging import Logger

# Record container start time immediately
CONTAINER_START_TIME = time.perf_counter()
CONTAINER_START_TIME_NS = time.time_ns()

# Import our telemetry module (after recording start time, hence E402)
from startup_telemetry import StartupContext, StartupTelemetry, init_startup_telemetry  # noqa: E402
from vllm.entrypoints.openai.api_server import run_server  # noqa: E402
from vllm.entrypoints.openai.cli_args import make_arg_parser  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402

logger = Logger("vllm_server_instrumented")

TIMEOUT_KEEP_ALIVE = 5  # seconds

# Global to track vLLM init timing
VLLM_INIT_START_TIME = None
VLLM_INIT_START_TIME_NS = None


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
        logger.warning(f"Error getting GPU memory: {e}")
        return None


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


def check_unknown_startup_memory_usage():
    """Check for unknown memory usage at startup."""
    gpu_free_memory = get_gpu_free_memory()
    if gpu_free_memory is not None:
        min_mem = min(gpu_free_memory)
        max_mem = max(gpu_free_memory)
        if max_mem - min_mem > 10:
            logger.warning(
                f"WARNING: Unbalanced GPU memory usage at start up. "
                f"This may cause OOM. Memory usage per GPU in MB: {gpu_free_memory}."
            )


def parse_args(parser: FlexibleArgumentParser):
    """Parse command line arguments."""
    parser = make_arg_parser(parser)
    parser.add_argument("--attention-backend", type=str, help="The attention backend to use")
    return parser.parse_args()


def health_check_loop(telemetry: StartupTelemetry, host: str = "localhost", port: int = 8000):
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

                    # Record vllm_init span as child of root pod_startup span
                    if VLLM_INIT_START_TIME_NS:
                        vllm_init_duration = ready_time - VLLM_INIT_START_TIME
                        telemetry.create_child_span(
                            "vllm_init",
                            start_time_ns=VLLM_INIT_START_TIME_NS,
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


async def run_instrumented_server(args):
    """Run vLLM server with startup instrumentation."""
    global VLLM_INIT_START_TIME, VLLM_INIT_START_TIME_NS

    # Initialize telemetry
    ctx = StartupContext(
        endpoint_name=os.environ.get("ENDPOINT_NAME", "poc-test"),
        model_name=os.environ.get(
            "MODEL_NAME", args.model if hasattr(args, "model") else "unknown"
        ),
        gpu_type=os.environ.get("GPU_TYPE", get_gpu_type()),
        num_gpus=int(os.environ.get("NUM_GPUS", "1")),
        region=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")),
        pod_uid=os.environ.get("POD_UID", "local-dev"),
        pod_name=os.environ.get("POD_NAME", "local-dev"),
        node_name=os.environ.get("NODE_NAME", "local-dev"),
    )

    telemetry = init_startup_telemetry(ctx)

    # Record download metrics if available (set by entrypoint.sh)
    download_duration_str = os.environ.get("STARTUP_DOWNLOAD_DURATION_S")
    download_size_mb_str = os.environ.get("STARTUP_DOWNLOAD_SIZE_MB")
    download_start_ts = os.environ.get("DOWNLOAD_START_TS")
    download_end_ts = os.environ.get("DOWNLOAD_END_TS")  # Used for python_init span

    if download_duration_str:
        download_duration = float(download_duration_str)
        download_size_mb = int(download_size_mb_str) if download_size_mb_str else 0

        # Create download span as child of root pod_startup span
        if download_start_ts and download_end_ts:
            start_ns = int(float(download_start_ts) * 1_000_000_000)
            end_ns = int(float(download_end_ts) * 1_000_000_000)

            telemetry.create_child_span(
                "s5cmd_download",
                start_time_ns=start_ns,
                end_time_ns=end_ns,
                attributes={
                    "download_size_mb": download_size_mb,
                    "duration_seconds": download_duration,
                },
            )

        telemetry.record_metric("download_duration", download_duration)
        print(
            f"[STARTUP METRICS] Recorded download metrics: {download_duration:.2f}s, {download_size_mb}MB"
        )

    # Mark when vLLM init starts (before creating python_init span)
    VLLM_INIT_START_TIME = time.perf_counter()
    VLLM_INIT_START_TIME_NS = time.time_ns()

    # Create python_init span to cover the gap between download end and vllm_init start
    # This includes: Python interpreter startup, heavy imports (vLLM, torch), arg parsing
    if download_end_ts:
        download_end_ns = int(float(download_end_ts) * 1_000_000_000)
        python_init_duration = (VLLM_INIT_START_TIME_NS - download_end_ns) / 1_000_000_000
        telemetry.create_child_span(
            "python_init",
            start_time_ns=download_end_ns,
            end_time_ns=VLLM_INIT_START_TIME_NS,
            attributes={
                "description": "Python startup, module imports, arg parsing",
                "duration_seconds": python_init_duration,
            },
        )
        print(f"[STARTUP METRICS] Python init (imports + setup): {python_init_duration:.2f}s")

    # Start health check thread to detect when server is ready
    port = getattr(args, "port", 8000)
    health_thread = threading.Thread(
        target=health_check_loop, args=(telemetry, "localhost", port), daemon=True
    )
    health_thread.start()

    # Start vLLM server (blocking)
    await run_server(args)


def main():
    """Main entry point."""
    print("[STARTUP METRICS] vLLM server starting...")

    # Check if download timing was captured by entrypoint.sh
    download_duration = os.environ.get("STARTUP_DOWNLOAD_DURATION_S")
    download_size_mb = os.environ.get("STARTUP_DOWNLOAD_SIZE_MB")
    if download_duration:
        print(f"[STARTUP METRICS] Model download took {download_duration}s ({download_size_mb}MB)")

    check_unknown_startup_memory_usage()

    parser = FlexibleArgumentParser()
    args = parse_args(parser)

    if args.attention_backend is not None:
        os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend

    # Run the instrumented server
    asyncio.run(run_instrumented_server(args))


if __name__ == "__main__":
    main()
