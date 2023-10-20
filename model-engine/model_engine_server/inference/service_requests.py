# Functions that help services (aka Servable instantiations) make requests to other Servable instantiations

import json
import os
from typing import Any, Callable, Dict, List
from uuid import uuid4

import boto3
import cloudpickle
from celery.result import allow_join_result
from model_engine_server.common.constants import DEFAULT_CELERY_TASK_NAME
from model_engine_server.common.errors import UpstreamHTTPSvcError
from model_engine_server.common.io import open_wrapper
from model_engine_server.common.service_requests import make_sync_request_with_retries
from model_engine_server.core.celery import TaskVisibility, celery_app
from model_engine_server.core.loggers import logger_name, make_logger

logger = make_logger(logger_name())

# TODO now that we're on SQS this won't work, since it connects to redis
s3_bucket: str = os.environ.get("CELERY_S3_BUCKET")  # type: ignore
celery = None


# Lazy initialization of Celery app in case we're running in test environments.
def get_celery():
    global celery
    if celery is None:
        celery = celery_app(
            None,
            aws_role=os.environ["AWS_PROFILE"],
            s3_bucket=s3_bucket,
            task_visibility=TaskVisibility.VISIBILITY_24H,
        )
    return celery


s3_client = None


# Lazy initialization of Celery app in case we're running in test environments.
def get_s3_client():
    global s3_client
    if s3_client is None:
        s3_client = boto3.client("s3", region_name="us-west-2")
    return s3_client


def _read_function_to_network_endpoint_info():
    # Dictionary format: {servable_id: {remote: true/false, endpoint_type: "sync"/"async", destination: <str>},...}
    # destination is either a celery queue name, i.e. launch.<something>, or the full url for an http request.
    details_json = os.getenv("CHILD_FN_INFO")
    if details_json is None:
        return None

    return json.loads(details_json)


child_fn_info = _read_function_to_network_endpoint_info()


def make_request(servable_id: str, local_fn: Callable, args: List[Any], kwargs: Dict[str, Any]):
    # This is the external-facing entrypoint. Reads in details and decides to make a network request or not
    # This function gets imported and called by the Launch client.
    current_fn_info = child_fn_info[servable_id]
    use_remote = current_fn_info["remote"]
    if use_remote:
        request_type = current_fn_info["endpoint_type"]
        destination = current_fn_info["destination"]
    else:
        request_type, destination = None, None
    if not use_remote:
        logger.info(f"Making local request to {servable_id}")
        return local_fn(*args, **kwargs)
    elif request_type == "sync":
        logger.info(f"Making sync network request to {servable_id}, {destination}")
        return _make_sync_request(destination, args, kwargs)
    elif request_type == "async":
        logger.info(f"Making async network request to {servable_id}, {destination}")
        return _make_async_request(destination, args, kwargs)
    # TODO eventually: add a switch for Triton here
    else:
        raise ValueError(
            f"current_fn_info is incorrect: needs valid endpoint_type + remote keys. Got {current_fn_info}"
        )


def _make_async_request(queue: str, args: List[Any], kwargs: Dict[str, Any]):
    # todo: possibly be able to make parallel requests by returning a Celery Result
    # e.g. by having a parallel=True and passing a list of args/kwargs.
    # request serialization: cloudpickle it and put it on s3, and put the request on s3
    request_body = _write_request(args, kwargs)
    res = get_celery().send_task(
        DEFAULT_CELERY_TASK_NAME,
        args=[dict(cloudpickle=request_body), True],
        queue=queue,
    )
    with allow_join_result():  # TODO this can cause deadlocks. Is the a way to identify it on this level?
        response = res.get()  # TODO this doesn't allow parallel requests, which we should allow
    # Empirically celery reraises the child exception
    return _read_response(response)


def _make_sync_request(request_url: str, args: List[Any], kwargs: Dict[str, Any]):
    request_body = _write_request(args, kwargs)
    try:
        response = make_sync_request_with_retries(
            request_url,
            payload_json=dict(cloudpickle=request_body, return_pickled=True),
        )
    except UpstreamHTTPSvcError as e:
        logger.error(
            f"Got upstream error from service {request_url}: status code {e.status_code}; body "
            f"{e.content!r}"
        )
        raise e
    return _read_response(response)


def _write_request(args: List[Any], kwargs: Dict[str, Any]):
    """
    Writes a network request argument object to s3

    Args:
        *args:
        **kwargs:

    Returns:
        s3location of file

    Side effect:
        A file corresponding to cloudpickle.dump({args: args, kwargs: kwargs}) is on s3 at the specified location
    """
    payload = dict(args=args, kwargs=kwargs)
    output_filename = f"{str(uuid4())}"
    temp_filename = f"/tmp/{output_filename}"
    with open_wrapper(temp_filename, "wb") as f:
        cloudpickle.dump(payload, f)

    # TODO change s3_key maybe?
    # For now stick intermediate results/inputs in same place we stick final results
    s3_bucket = os.environ["RESULTS_S3_BUCKET"]
    s3_key = f"tmp/hosted-model-inference-intermediate-inputs/{output_filename}"
    with open(temp_filename, "rb") as f:
        get_s3_client().upload_fileobj(f, s3_bucket, s3_key)
    payload_location = os.path.join(f"s3://{s3_bucket}", s3_key)
    return payload_location


def _read_response(response):
    """
    Reads a response from another service from s3.
    Keep in line with _write_to_s3 inside of inference/common

    Args:
        response: The response object from the other service.

    Returns:

    """
    # If we get here, response should have "result_url", since otherwise the request should have failed.
    result_location = response["result_url"]
    with open_wrapper(result_location, "rb") as f:
        return cloudpickle.load(f)
