from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests
from model_engine_server.common.constants import (
    BILLING_POST_INFERENCE_HOOK,
    CALLBACK_POST_INFERENCE_HOOK,
)
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from model_engine_server.core.loggers import filename_wo_ext, make_logger
from model_engine_server.domain.entities import CallbackAuth, CallbackBasicAuth
from model_engine_server.inference.common import _write_to_s3
from model_engine_server.inference.domain.gateways.inference_monitoring_metrics_gateway import (
    InferenceMonitoringMetricsGateway,
)
from model_engine_server.inference.domain.gateways.usage_metrics_gateway import UsageMetricsGateway
from model_engine_server.inference.infra.gateways.fake_usage_metrics_gateway import (
    FakeUsageMetricsGateway,
)
from tenacity import Retrying, stop_after_attempt, wait_exponential

logger = make_logger(filename_wo_ext(__file__))


def _upload_data(data: Any):
    return _write_to_s3(data).get("result_url")


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


class BillingHook(PostInferenceHook):
    def __init__(
        self,
        endpoint_name: str,
        bundle_name: str,
        user_id: str,
        billing_queue: Optional[str],
        billing_tags: Optional[Dict[str, Any]],
    ):
        super().__init__(endpoint_name, bundle_name, user_id)
        self._billing_queue = billing_queue
        self._billing_tags = billing_tags or {}

    def handle(
        self,
        request_payload: EndpointPredictV1Request,
        response: Dict[str, Any],
        task_id: Optional[str],
    ):
        if not self._user_id or not self._billing_queue:
            logger.error("Usage inputs could not be found for billing hook, aborting")
            return
        if not task_id:
            task_id = str(uuid4())

        events_queue: UsageMetricsGateway
        try:
            from plugins.eventbridge_usage_metrics_gateway import EventbridgeUsageMetricsGateway

            events_queue = EventbridgeUsageMetricsGateway(self._billing_queue)
        except ModuleNotFoundError:
            events_queue = FakeUsageMetricsGateway()
        events_queue.emit_task_call_metric(idempotency_token=task_id, tags=self._billing_tags)


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
                if hook_lower == BILLING_POST_INFERENCE_HOOK:
                    self._hooks[hook_lower] = BillingHook(
                        endpoint_name,
                        bundle_name,
                        user_id,
                        billing_queue,
                        billing_tags,
                    )
                elif hook_lower == CALLBACK_POST_INFERENCE_HOOK:
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
        response: Dict[str, Any],
        task_id: Optional[str] = None,
    ):
        for hook_name, hook in self._hooks.items():
            self._monitoring_metrics_gateway.emit_attempted_post_inference_hook(hook_name)
            try:
                hook.handle(request_payload, response, task_id)
                self._monitoring_metrics_gateway.emit_successful_post_inference_hook(hook_name)
            except Exception:
                logger.exception(f"Hook {hook_name} failed.")
