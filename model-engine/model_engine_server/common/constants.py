from pathlib import Path

BILLING_POST_INFERENCE_HOOK: str = "billing"
CALLBACK_POST_INFERENCE_HOOK: str = "callback"
LOGGING_POST_INFERENCE_HOOK: str = "logging"
SUPPORTED_POST_INFERENCE_HOOKS: list = [
    BILLING_POST_INFERENCE_HOOK,
    CALLBACK_POST_INFERENCE_HOOK,
    LOGGING_POST_INFERENCE_HOOK,
]
READYZ_FPATH: str = "/tmp/readyz"
DEFAULT_CELERY_TASK_NAME: str = "hosted_model_inference.inference.async_inference.tasks.predict"
LIRA_CELERY_TASK_NAME: str = "ml_serve.celery_service.exec_func"

# Endpoint garbage collection state kept in endpoint_metadata. Reserved: a metadata update that
# omits them keeps the stored values, so an opt-out or a running clock survives owner edits.
ENDPOINT_GC_UNAVAILABLE_SINCE_KEY: str = "_gc_unavailable_since"
ENDPOINT_GC_LAST_TRAFFIC_AT_KEY: str = "_gc_last_traffic_at"
ENDPOINT_GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: str = "_gc_scale_to_zero_requested_at"
ENDPOINT_GC_TOUCHED_AT_KEY: str = "_gc_touched_at"
ENDPOINT_GC_EXEMPT_KEY: str = "_gc_exempt"
ENDPOINT_GC_METADATA_KEYS: tuple = (
    ENDPOINT_GC_UNAVAILABLE_SINCE_KEY,
    ENDPOINT_GC_LAST_TRAFFIC_AT_KEY,
    ENDPOINT_GC_SCALE_TO_ZERO_REQUESTED_AT_KEY,
    ENDPOINT_GC_TOUCHED_AT_KEY,
    ENDPOINT_GC_EXEMPT_KEY,
)

PROJECT_ROOT: Path = Path(__file__).parents[2].absolute()
HOSTED_MODEL_INFERENCE_ROOT: Path = PROJECT_ROOT / "model-engine"
