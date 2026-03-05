import os
import time

# Capture Python start time BEFORE heavy imports (for python_init metric)
_PYTHON_START_TIME = time.perf_counter()

# Now do heavy imports (noqa: E402 - intentional late import for startup time measurement)
import asyncio  # noqa: E402

from vllm.entrypoints.openai.cli_args import make_arg_parser  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402
from vllm_omni.entrypoints.openai.api_server import omni_run_server  # noqa: E402

from .utils.resource_debug import check_unknown_startup_memory_usage  # noqa: E402
from .utils.startup_telemetry import with_startup_metrics  # noqa: E402

if __name__ == "__main__":
    check_unknown_startup_memory_usage()

    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args()
    if args.attention_backend is not None:
        os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend

    asyncio.run(
        with_startup_metrics(omni_run_server, args=args, python_start_time=_PYTHON_START_TIME)
    )
