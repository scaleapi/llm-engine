import importlib
import os
import shutil

from model_engine_server.core.loggers import make_logger

logger = make_logger(__name__)

LOCAL_BUNDLE_PATH = os.getenv("LOCAL_BUNDLE_PATH", "")
LOAD_MODEL_MODULE_PATH = os.getenv("LOAD_MODEL_MODULE_PATH", "")
LOAD_PREDICT_MODULE_PATH = os.getenv("LOAD_PREDICT_MODULE_PATH", "")

# Matches the user endpoint docker images, specifically where code gets copied
# and where the user actually owns the files
BASE_PATH_IN_ENDPOINT = "/app"


def load_module(module_path):
    parts = module_path.split(".")
    module_path_list, fn_name = parts[:-1], parts[-1]
    module_path = ".".join(module_path_list)
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


def download_and_inject_bundle():
    logger.info(f"Unzipping bundle from location {LOCAL_BUNDLE_PATH} to {BASE_PATH_IN_ENDPOINT}")
    shutil.unpack_archive(LOCAL_BUNDLE_PATH, BASE_PATH_IN_ENDPOINT, "zip")

    # Load in functions to create *.pyc files in the __pycache__ in the image which
    # should make loading faster at worker startup time
    # TODO(sh763059): renable this if it becomes a bottleneck
    # load_model_fn = load_module(LOAD_MODEL_MODULE_PATH)
    # load_predict_fn = load_module(LOAD_PREDICT_MODULE_PATH)

    # Clean up serialized bundle file to save storage
    if os.path.exists(LOCAL_BUNDLE_PATH):
        os.remove(LOCAL_BUNDLE_PATH)
    else:
        logger.error(f"No bundle found at {LOCAL_BUNDLE_PATH}!")


if __name__ == "__main__":
    download_and_inject_bundle()
