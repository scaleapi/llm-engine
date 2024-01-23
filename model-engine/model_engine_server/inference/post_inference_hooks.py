import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import requests
from fastapi.responses import JSONResponse
from model_engine_server.common.constants import CALLBACK_POST_INFERENCE_HOOK
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.entities import CallbackAuth, CallbackBasicAuth
from model_engine_server.inference.domain.gateways.inference_monitoring_metrics_gateway import (
    InferenceMonitoringMetricsGateway,
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
                hook.handle(request_payload, loaded_response, task_id)
                self._monitoring_metrics_gateway.emit_successful_post_inference_hook(hook_name)
            except Exception:
                logger.exception(f"Hook {hook_name} failed.")
