from uvicorn.workers import UvicornWorker

CONCURRENCY_LIMIT = 1000


class LaunchWorker(UvicornWorker):
    """Overrides the configuration of the Uvicorn Worker."""

    # uvloop and httptools are both faster than their alternatives, but they are not compatible
    # with Windows or PyPy.
    CONFIG_KWARGS = {"loop": "uvloop", "http": "httptools", "limit_concurrency": CONCURRENCY_LIMIT}
