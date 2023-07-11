import importlib
import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Dict
from uuid import uuid4

import boto3
import cloudpickle

from spellbook_serve.common.dtos.tasks import EndpointPredictV1Request, RequestSchema
from spellbook_serve.common.io import open_wrapper
from spellbook_serve.common.serialization_utils import b64_to_python_json
from spellbook_serve.core.loggers import make_logger
from spellbook_serve.core.utils.timer import timer
from spellbook_serve.domain.entities import ModelEndpointConfig
from spellbook_serve.inference.service_requests import make_request

logger = make_logger(__name__)

s3_client = None


def get_s3_client():
    global s3_client
    if s3_client is None:
        s3_client = boto3.client("s3", region_name="us-west-2")
    return s3_client


# Required environment variables
# TODO: clean up environment variables with perhaps a config structure / file.
BUNDLE_URL_KEY = "BUNDLE_URL"
BASE_PATH_KEY = "BASE_PATH"
RESULTS_S3_BUCKET_KEY = "RESULTS_S3_BUCKET"

# Optional environment variables
LOAD_PREDICT_FN_MODULE_PATH_KEY = "LOAD_PREDICT_FN_MODULE_PATH"
LOAD_MODEL_FN_MODULE_PATH_KEY = "LOAD_MODEL_FN_MODULE_PATH"
USER_CONFIG_LOCATION_KEY = "USER_CONFIG_LOCATION"
ENDPOINT_CONFIG_LOCATION_KEY = "ENDPOINT_CONFIG_LOCATION"
LOCAL_BUNDLE_PATH_KEY = "LOCAL_BUNDLE_PATH"


def _load_fn_from_module(full_module_path: str) -> Callable:
    parts = full_module_path.split(".")
    module_path_list, fn_name = parts[:-1], parts[-1]
    module_path = ".".join(module_path_list)
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


def _set_make_request_fn(obj: Any) -> None:
    if hasattr(obj, "set_make_request_fn"):
        # When a bundle inherits from a ServiceDescription class
        # it means it can be a part of the pipeline.
        # We want to give it an option to call other services.
        obj.set_make_request_fn(make_request)


def load_predict_fn_or_cls():
    bundle_url = os.getenv(BUNDLE_URL_KEY)
    load_predict_fn_module_path = os.getenv(LOAD_PREDICT_FN_MODULE_PATH_KEY, "")
    load_model_fn_module_path = os.getenv(LOAD_MODEL_FN_MODULE_PATH_KEY, "")
    local_bundle_path = os.getenv(LOCAL_BUNDLE_PATH_KEY, "")
    base_path = os.getenv(BASE_PATH_KEY)

    assert bundle_url is not None
    assert base_path is not None
    default_user_config_location = os.path.join(
        base_path, "user_config"
    )  # Keep in sync with the volumeMount in the deployment yaml
    user_config_location = os.getenv(USER_CONFIG_LOCATION_KEY, default_user_config_location)
    with open_wrapper(user_config_location, "r") as f:
        b64text = f.read().strip("\n")
        deserialized_config = b64_to_python_json(b64text)
        logger.info(f"Read in config {deserialized_config}")

        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(
            "Current files in directory:\n"
            f"{subprocess.check_output(['ls', '-l', '.']).decode('utf-8')}"
        )

    if load_predict_fn_module_path and load_model_fn_module_path:
        if local_bundle_path:
            logger.info("Bundle has already been loaded into container")
        else:
            logger.info(f"Loading bundle from inside the container {bundle_url}")

            with tempfile.TemporaryDirectory() as tmpdir:
                local_zip_path = os.path.join(tmpdir, "bundle.zip")
                with timer(logger=logger, name="download_zip_bundle"):
                    with open_wrapper(bundle_url, "rb") as remote_zip_f, open(
                        local_zip_path, "wb"
                    ) as local_zip_f:
                        local_zip_f.write(remote_zip_f.read())

                with timer(logger=logger, name="unzip_bundle"):
                    shutil.unpack_archive(local_zip_path, base_path, "zip")
                    # TODO might be bugged with some zip files? I tried absolute paths and it failed for me

        with timer(logger=logger, name="load_model_fn_from_module"):
            load_model_fn = _load_fn_from_module(load_model_fn_module_path)

        with timer(logger=logger, name="load_predict_fn_from_module"):
            load_predict_fn_inner = _load_fn_from_module(load_predict_fn_module_path)

        with timer(logger=logger, name="load_model_fn"):
            model = load_model_fn(deserialized_config)

        with timer(logger=logger, name="load_predict_fn_inner"):
            predict_fn_inner = load_predict_fn_inner(deserialized_config, model)

        _set_make_request_fn(predict_fn_inner)
        return predict_fn_inner
    else:
        logger.info("Loading bundle from serialized object")
        # e.g. s3://scale-ml/hosted-model-inference/predict_fns/abc123

        with timer(logger=logger, name="download_and_deserialize_cloudpickle_bundle"):
            with open_wrapper(bundle_url, "rb") as f:
                with timer(logger=logger, name="deserialize_cloudpickle_bundle"):
                    bundle = cloudpickle.load(f)

        if isinstance(bundle, dict):
            # LEGACY: bundle defined by a dictionary
            # TODO: validate versions of python, pytorch, etc.

            if "model" in bundle:
                model = bundle["model"]
            elif "load_model_fn" in bundle:
                # e.g. s3://scale-ml/hosted-model-inference/tf-saved-models/tf-cpu-efficientdet-abc123.tar.gz
                with timer(logger=logger, name="load_model_fn"):
                    if deserialized_config is None:
                        model = bundle["load_model_fn"]()
                    else:
                        model = bundle["load_model_fn"](deserialized_config)
            else:
                raise ValueError("Need to specify either 'model' or 'load_model_fn' in the bundle")

            load_predict_fn_inner = bundle["load_predict_fn"]

            with timer(logger=logger, name="load_predict_fn_inner"):
                if deserialized_config is None:
                    predict_fn_inner = load_predict_fn_inner(model)
                else:
                    predict_fn_inner = load_predict_fn_inner(deserialized_config, model)

            _set_make_request_fn(predict_fn_inner)
            return predict_fn_inner
        elif hasattr(bundle, "init"):
            # if the bundle needs initialization on the server side, call init() method
            # TODO (ivan) support args/kwargs defined on "create_bundle/endpoint" step
            # Does passing in the config dictionary satisfy passing args/kwargs?
            bundle.init(deserialized_config)
        _set_make_request_fn(bundle)
        return bundle


def _write_to_s3(output: Any) -> Dict[str, str]:
    uuid_ = str(uuid4())
    output_filename = f"{uuid_}.pkl"
    temp_filename = f"/tmp/{output_filename}"
    with open(temp_filename, "wb") as f:
        cloudpickle.dump(output, f)

    # TODO change s3_key maybe?
    s3_bucket = os.getenv(RESULTS_S3_BUCKET_KEY)
    assert s3_bucket is not None
    s3_key = f"tmp/hosted-model-inference-outputs/{output_filename}"
    with open(temp_filename, "rb") as f:
        get_s3_client().upload_fileobj(f, s3_bucket, s3_key)

    os.remove(temp_filename)

    result_uri = os.path.join(f"s3://{s3_bucket}", s3_key)
    return {"result_url": result_uri}


def serialize_json_result(result):
    return {"result": json.dumps(result)}


def deserialize_json_result(result_serialized):
    return json.loads(result_serialized["result"])


def predict_on_url(predict_fn: Callable, request_url: str, return_pickled: bool) -> Dict[str, str]:
    with open_wrapper(request_url, "rb") as f:
        output = predict_fn(f.read())

        if return_pickled:
            return _write_to_s3(output)
        else:
            return serialize_json_result(output)


def predict_on_args(
    predict_fn: Callable, inputs: RequestSchema, return_pickled: bool
) -> Dict[str, str]:
    inputs_kwargs = inputs.__root__
    output = predict_fn(**inputs_kwargs)

    if return_pickled:
        return _write_to_s3(output)
    else:
        return serialize_json_result(output)


def predict_on_cloudpickle(predict_fn: Callable, input_location: str) -> Dict[str, str]:
    """
    Run predict_fn on a cloudpickled payload. Should be used only by intermediate stages in a pipeline,
        not called directly by the Gateway. Will always return a cloudpickle (for now).
    Args:
        predict_fn:
        input_location: s3url of cloudpickled arguments

    Returns:

    """
    # TODO we currently always write our output to s3, as opposed to serializing it.
    #  We can expose these switches if necessary, although this complicates the user api.
    with open_wrapper(input_location, "rb") as f:
        inputs = cloudpickle.load(f)
        output = predict_fn(*inputs["args"], **inputs["kwargs"])
        return _write_to_s3(output)


def run_predict(predict_fn: Callable, request_params: EndpointPredictV1Request) -> Dict[str, str]:
    """

    Args:
        predict_fn:
        request_params: A dictionary containing either "url" or "args" or "cloudpickle",
            corresponding to a file at a url, or a list of arguments, or a cloudpickled dictionary
            {"args": [arg1, arg2], "kwargs": {"kwarg1": val1, "kwarg2": val2}}
        return_pickled:

    Returns:

    """
    if request_params.url is not None:
        return predict_on_url(predict_fn, request_params.url, request_params.return_pickled)
    elif request_params.args is not None:
        return predict_on_args(predict_fn, request_params.args, request_params.return_pickled)
    elif request_params.cloudpickle is not None:
        return predict_on_cloudpickle(predict_fn, request_params.cloudpickle)
    else:
        raise ValueError("Input needs either url or args.")


def get_endpoint_config():
    base_path = os.getenv(BASE_PATH_KEY)
    assert base_path is not None

    default_endpoint_config_location = os.path.join(
        base_path, "endpoint_config"
    )  # Keep in sync with the volumeMount in the deployment yaml
    endpoint_config_location = os.getenv(
        ENDPOINT_CONFIG_LOCATION_KEY, default_endpoint_config_location
    )
    with open_wrapper(endpoint_config_location, "r") as f:
        b64text = f.read().strip("\n")
    endpoint_config = ModelEndpointConfig.deserialize(b64text)
    return endpoint_config


def is_sensitive_envvar(var):
    return var.startswith("LAUNCH_") or var.startswith("HMI_")


def unset_sensitive_envvars():
    # Since all the pods are in the same namespace as of now, there are env vars e.g.
    # `LAUNCH_<USER_ID>_...` that store the IPs of various services (and also leak that the services exist)
    # Let's unset them here
    # The names seem to be the name of the deployment, which always starts with `LAUNCH_` or `HMI_`.
    logger.info("Unsetting environment variables...")
    sensitive_envvars = [var for var in os.environ if is_sensitive_envvar(var)]
    for var in sensitive_envvars:
        del os.environ[var]
