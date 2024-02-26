import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pytz
import requests
from fastapi.responses import JSONResponse
from model_engine_server.common.constants import (
    CALLBACK_POST_INFERENCE_HOOK,
    LOGGING_POST_INFERENCE_HOOK,
)
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.entities import CallbackAuth, CallbackBasicAuth
from model_engine_server.domain.entities.model_endpoint_entity import ModelEndpointType
from model_engine_server.domain.exceptions import StreamPutException
from model_engine_server.inference.domain.gateways.inference_monitoring_metrics_gateway import (
    InferenceMonitoringMetricsGateway,
)
from model_engine_server.inference.domain.gateways.streaming_storage_gateway import (
    StreamingStorageGateway,
)
from tenacity import Retrying, stop_after_attempt, wait_exponential

logger = make_logger(logger_name())


class PostInferenceHook(ABC):
    def __init__(
        self,
        endpoint_name: str,
        bundle_name: str,
        user_id: str,
    ):
        self._endpoint_name = endpoint_name
        self._bundle_name = bundle_name
        self._user_id = user_id

    @abstractmethod
    def handle(
        self,
        request_payload: EndpointPredictV1Request,
        response: Dict[str, Any],
        task_id: Optional[str],
    ):
        pass


class CallbackHook(PostInferenceHook):
    def __init__(
        self,
        endpoint_name: str,
        bundle_name: str,
        user_id: str,
        default_callback_url: Optional[str],
        default_callback_auth: Optional[CallbackAuth],
    ):
        super().__init__(endpoint_name, bundle_name, user_id)
        self._default_callback_url = default_callback_url
        self._default_callback_auth = default_callback_auth

    def handle(
        self,
        request_payload: EndpointPredictV1Request,
        response: Dict[str, Any],
        task_id: Optional[str],
    ):
        logger.info(f"Handling a callback hook for endpoint {self._endpoint_name}.")
        callback_url = request_payload.callback_url
        if not callback_url:
            callback_url = self._default_callback_url
        if not callback_url:
            logger.warning("No callback URL specified for request.")
            return

        response["task_id"] = task_id
        auth = request_payload.callback_auth or self._default_callback_auth
        if auth and isinstance(auth.__root__, CallbackBasicAuth):
            auth_tuple = (auth.__root__.username, auth.__root__.password)
        else:
            auth_tuple = (self._user_id, "")

        for attempt in Retrying(stop=stop_after_attempt(3), wait=wait_exponential()):
            with attempt:
                res = requests.post(url=callback_url, json=response, auth=auth_tuple)
                assert 200 <= res.status_code < 300


class LoggingHook(PostInferenceHook):
    def __init__(
        self,
        endpoint_name: str,
        bundle_name: str,
        user_id: str,
        endpoint_id: Optional[str],
        endpoint_type: Optional[ModelEndpointType],
        bundle_id: Optional[str],
        labels: Optional[Dict[str, str]],
        streaming_storage_gateway: StreamingStorageGateway,
    ):
        super().__init__(endpoint_name, bundle_name, user_id)
        self._endpoint_id = endpoint_id
        self._endpoint_type = endpoint_type
        self._bundle_id = bundle_id
        self._labels = labels
        self._streaming_storage_gateway = streaming_storage_gateway

    def handle(
        self,
        request_payload: EndpointPredictV1Request,
        response: Dict[str, Any],
        task_id: Optional[str],
    ):
        logger.info(f"Handling a logging hook for endpoint {self._endpoint_name}.")
        if (
            not self._endpoint_id
            or not self._endpoint_type
            or not self._bundle_id
            or not self._labels
        ):
            logger.warning(
                "No endpoint_id, endpoint_type, bundle_id, or labels specified for request."
            )
            return
        response["task_id"] = task_id
        data_record = {
            "EMITTED_AT": datetime.now(pytz.timezone("UTC")).strftime("%Y-%m-%dT%H:%M:%S"),
            "REQUEST_BODY": request_payload.json(),
            "RESPONSE_BODY": response,
            "ENDPOINT_ID": self._endpoint_id,
            "ENDPOINT_NAME": self._endpoint_name,
            "ENDPOINT_TYPE": self._endpoint_type.value,
            "BUNDLE_ID": self._bundle_id,
            "LABELS": self._labels,
        }
        stream_name = infra_config().firehose_stream_name
        if stream_name is None:
            logger.warning("No firehose stream name specified. Logging hook will not be executed.")
            return
        streaming_storage_response = {}  # pragma: no cover
        try:
            streaming_storage_response = (
                self._streaming_storage_gateway.put_record(  # pragma: no cover
                    stream_name=stream_name, record=data_record
                )
            )
        except StreamPutException:  # pragma: no cover
            logger.error(  # pragma: no cover
                f"Failed to put record into firehose stream {stream_name}. Response metadata {streaming_storage_response.get('ResponseMetadata')}."
            )


class PostInferenceHooksHandler:
    def __init__(
        self,
        endpoint_name: str,
        bundle_name: str,
        user_id: str,
        billing_queue: str,
        billing_tags: Dict[str, Any],
        default_callback_url: Optional[str],
        default_callback_auth: Optional[CallbackAuth],
        post_inference_hooks: Optional[List[str]],
        monitoring_metrics_gateway: InferenceMonitoringMetricsGateway,
        endpoint_id: Optional[str],
        endpoint_type: Optional[ModelEndpointType],
        bundle_id: Optional[str],
        labels: Optional[Dict[str, str]],
        streaming_storage_gateway: StreamingStorageGateway,
    ):
        self._monitoring_metrics_gateway = monitoring_metrics_gateway
        self._hooks: Dict[str, PostInferenceHook] = {}
        if post_inference_hooks:
            for hook in post_inference_hooks:
                # TODO: Ensure that this process gracefully handles errors in
                #   initializing each post-inference hook.
                hook_lower = hook.lower()
                if hook_lower == CALLBACK_POST_INFERENCE_HOOK:
                    self._hooks[hook_lower] = CallbackHook(
                        endpoint_name,
                        bundle_name,
                        user_id,
                        default_callback_url,
                        default_callback_auth,
                    )
                elif hook_lower == LOGGING_POST_INFERENCE_HOOK:
                    self._hooks[hook_lower] = LoggingHook(
                        endpoint_name,
                        bundle_name,
                        user_id,
                        endpoint_id,
                        endpoint_type,
                        bundle_id,
                        labels,
                        streaming_storage_gateway,
                    )
                else:
                    raise ValueError(f"Hook {hook_lower} is currently not supported.")

    def handle(
        self,
        request_payload: EndpointPredictV1Request,
        response: Union[Dict[str, Any], JSONResponse],
        task_id: Optional[str] = None,
    ):
        if isinstance(response, JSONResponse):
            loaded_response = json.loads(response.body)
        else:
            loaded_response = response
        for hook_name, hook in self._hooks.items():
            self._monitoring_metrics_gateway.emit_attempted_post_inference_hook(hook_name)
            try:
                hook.handle(request_payload, loaded_response, task_id)  # pragma: no cover
                self._monitoring_metrics_gateway.emit_successful_post_inference_hook(hook_name)
            except Exception:
                logger.exception(f"Hook {hook_name} failed.")
