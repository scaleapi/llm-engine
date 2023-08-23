from pathlib import Path

BILLING_POST_INFERENCE_HOOK: str = "billing"
CALLBACK_POST_INFERENCE_HOOK: str = "callback"
READYZ_FPATH: str = "/tmp/readyz"
DEFAULT_CELERY_TASK_NAME: str = "hosted_model_inference.inference.async_inference.tasks.predict"
LIRA_CELERY_TASK_NAME: str = "ml_serve.celery_service.exec_func"

PROJECT_ROOT: Path = Path(__file__).parents[2].absolute()
HOSTED_MODEL_INFERENCE_ROOT: Path = PROJECT_ROOT / "model-engine"
