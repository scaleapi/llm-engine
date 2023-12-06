from uvicorn.workers import UvicornWorker

# Gunicorn returns 503 instead of 429 when concurrency exceeds the limit, before adding rate limiting just increase the concurrency
# We'll autoscale at target concurrency of a much lower number (around 50), and this just makes sure we don't 503 with bursty traffic
CONCURRENCY_LIMIT = 10000


class LaunchWorker(UvicornWorker):
    """Overrides the configuration of the Uvicorn Worker."""

    # uvloop and httptools are both faster than their alternatives, but they are not compatible
    # with Windows or PyPy.
    CONFIG_KWARGS = {"loop": "uvloop", "http": "httptools", "limit_concurrency": CONCURRENCY_LIMIT}
