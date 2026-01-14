"""
vLLM Startup Wrapper with Download Metrics.

This wrapper script handles model download and emits the download span/metric,
then execs into vllm_server which handles python_init and vllm_init metrics.

Usage:
    DOWNLOAD_CMD="s5cmd ..." python -m vllm_startup_wrapper --model model_files ...

Environment variables:
    DOWNLOAD_CMD: Shell command to download model files (required)
    ENABLE_STARTUP_METRICS: Set to "true" to enable metrics emission
    POD_UID: Kubernetes pod UID for trace correlation
    POD_NAME: Pod name for metrics tagging
    ENDPOINT_NAME: Endpoint name for metrics tagging
    MODEL_NAME: Model name for metrics tagging
"""

import os
import subprocess
import sys
import time

# Feature gate
ENABLE_STARTUP_METRICS = os.environ.get("ENABLE_STARTUP_METRICS", "").lower() == "true"


def get_download_size_mb(directory: str) -> int:
    """Get total size of downloaded files in MB."""
    try:
        result = subprocess.run(
            ["du", "-sm", directory],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.split()[0])
    except Exception:
        return 0


def run_download(download_cmd: str) -> dict:
    """Run download command and return timing info."""
    print("[WRAPPER] Running download...")

    start_time_ns = time.time_ns()
    start_perf = time.perf_counter()

    subprocess.run(download_cmd, shell=True, check=True)

    end_time_ns = time.time_ns()
    duration = time.perf_counter() - start_perf

    # Get download size
    model_dir = "mistral_files" if os.path.exists("mistral_files") else "model_files"
    size_mb = get_download_size_mb(model_dir)

    print(f"[WRAPPER] Download completed: {duration:.2f}s ({size_mb}MB)")

    return {
        "start_time_ns": start_time_ns,
        "end_time_ns": end_time_ns,
        "duration_s": duration,
        "size_mb": size_mb,
    }


def emit_download_metrics(timing: dict):
    """Emit download span and metric via OTel."""
    from startup_telemetry import StartupContext, init_startup_telemetry

    ctx = StartupContext(
        endpoint_name=os.environ.get("ENDPOINT_NAME", "unknown"),
        model_name=os.environ.get("MODEL_NAME", "unknown"),
        gpu_type=os.environ.get("GPU_TYPE", "unknown"),
        num_gpus=int(os.environ.get("NUM_GPUS", "1")),
        region=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")),
        pod_uid=os.environ.get("POD_UID", "unknown"),
        pod_name=os.environ.get("POD_NAME", "unknown"),
        node_name=os.environ.get("NODE_NAME", "unknown"),
    )

    telemetry = init_startup_telemetry(ctx)

    # Emit download span
    telemetry.create_child_span(
        "s3_download",
        start_time_ns=timing["start_time_ns"],
        end_time_ns=timing["end_time_ns"],
        attributes={
            "download_size_mb": timing["size_mb"],
            "duration_seconds": timing["duration_s"],
        },
    )

    # Emit download metric
    telemetry.record_metric("download_duration", timing["duration_s"])

    # Flush before exec (important!)
    telemetry.flush()
    print("[WRAPPER] Download metrics emitted")


def main():
    """Main entry point."""
    download_cmd = os.environ.get("DOWNLOAD_CMD")

    if download_cmd:
        timing = run_download(download_cmd)

        if ENABLE_STARTUP_METRICS:
            emit_download_metrics(timing)
    else:
        print("[WRAPPER] No DOWNLOAD_CMD, skipping download")

    # Exec into vllm_server
    server_args = sys.argv[1:]
    python_exe = sys.executable

    print("[WRAPPER] Starting vllm_server...")
    os.execvp(python_exe, [python_exe, "-m", "vllm_server"] + server_args)


if __name__ == "__main__":
    main()
