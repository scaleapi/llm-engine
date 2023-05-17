from uvicorn.workers import UvicornWorker

# The target concurrency is around 50, so we set the limit to 32 with 4 workers
# for a total concurrency of 128 to allow for some headroom.
CONCURRENCY_LIMIT = 32


class LLMEngineWorker(UvicornWorker):
    """Overrides the configuration of the Uvicorn Worker."""

    # uvloop and httptools are both faster than their alternatives, but they are not compatible
    # with Windows or PyPy.
    CONFIG_KWARGS = {
        "loop": "uvloop",
        "http": "httptools",
        "limit_concurrency": CONCURRENCY_LIMIT,
    }
