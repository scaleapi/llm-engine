from pathlib import Path

CALLBACK_POST_INFERENCE_HOOK: str = "callback"
READYZ_FPATH: str = "/tmp/readyz"
DEFAULT_CELERY_TASK_NAME: str = "spellbook_serve.inference.async_inference.tasks.predict"
LIRA_CELERY_TASK_NAME: str = "spellbook_serve.inference.celery_service.exec_func"  # TODO: FIXME

PROJECT_ROOT: Path = Path(__file__).parents[2].absolute()
HOSTED_MODEL_INFERENCE_ROOT: Path = PROJECT_ROOT / "spellbook_serve"

FEATURE_FLAG_USE_MULTI_CONTAINER_ARCHITECTURE_FOR_ARTIFACTLIKE_BUNDLE: str = (
    "USE_MULTI_CONTAINER_ARCHITECTURE_FOR_ARTIFACTLIKE_BUNDLE"
)
