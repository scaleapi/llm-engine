# Represents the infrastructure underlying a model endpoint.
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from model_engine_server.domain.entities import (
    CallbackAuth,
    CpuSpecificationType,
    GpuType,
    ModelEndpointInfraState,
    ModelEndpointRecord,
    StorageSpecificationType,
)


class ModelEndpointInfraGateway(ABC):
    """
    Base class for Model Endpoint Infra gateways.
    """

    @abstractmethod
    def create_model_endpoint_infra(
        self,
        *,
        model_endpoint_record: ModelEndpointRecord,
        min_workers: int,
        max_workers: int,
        per_worker: int,
        concurrent_requests: int,
        cpus: CpuSpecificationType,
        gpus: int,
        memory: StorageSpecificationType,
        gpu_type: Optional[GpuType],
        storage: StorageSpecificationType,
        nodes_per_worker: int,
        optimize_costs: bool,
        aws_role: str,
        results_s3_bucket: str,
        child_fn_info: Optional[Dict[str, Any]],
        post_inference_hooks: Optional[List[str]],
        labels: Dict[str, str],
        prewarm: bool,
        high_priority: Optional[bool],
        billing_tags: Optional[Dict[str, Any]] = None,
        default_callback_url: Optional[str],
        default_callback_auth: Optional[CallbackAuth],
    ) -> str:
        """
        Creates the underlying infrastructure for a Model Endpoint.

        Args:
            model_endpoint_record: The associated record of a model endpoint.
            min_workers: The minimum number of workers for the model endpoint.
            max_workers: The maximum number of workers for the model endpoint.
            per_worker: Autoscaling parameter used to set the target number of enqueued items per worker.
            concurrent_requests: The max number of concurrent requests for the worker to work on.
            cpus: The amount of CPU to use per worker for the model endpoint.
            gpus: The amount of GPU to use per worker for the model endpoint.
            memory: The amount of memory to use per worker for the model endpoint.
            gpu_type: The type of GPU to use per worker for the model endpoint.
            storage: The amount of storage to request per worker for the model endpoint.
            optimize_costs: Whether to automatically infer CPU and memory.
            aws_role: The AWS role to use.
            results_s3_bucket: The S3 bucket to store results.
            child_fn_info: For pipelines.
            post_inference_hooks: A list of optional post-inference hooks to perform.
            labels: Labels to attach to the infrastructure for tracking purposes.
            prewarm: For async endpoints only, whether to have pods load resources on worker startup
                as opposed to on first request. If loading takes more than 5 minutes then set this
                to False
            high_priority: Makes all pods for this endpoint higher priority to enable faster pod spinup
                time. Higher priority pods will displace the lower priority dummy pods from shared pool.
            billing_tags: Arbitrary tags passed to billing
            default_callback_url: The default callback URL to use for the model endpoint.
            default_callback_auth: The default callback auth to use for the model endpoint.

        Returns:
            A unique ID for the task to create the infrastructure resources.
        """

    @abstractmethod
    async def update_model_endpoint_infra(
        self,
        *,
        model_endpoint_record: ModelEndpointRecord,
        min_workers: Optional[int] = None,
        max_workers: Optional[int] = None,
        per_worker: Optional[int] = None,
        concurrent_requests: Optional[int] = None,
        cpus: Optional[CpuSpecificationType] = None,
        gpus: Optional[int] = None,
        memory: Optional[StorageSpecificationType] = None,
        gpu_type: Optional[GpuType] = None,
        storage: Optional[StorageSpecificationType] = None,
        optimize_costs: Optional[bool] = None,
        child_fn_info: Optional[Dict[str, Any]] = None,
        post_inference_hooks: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        prewarm: Optional[bool] = None,
        high_priority: Optional[bool] = None,
        billing_tags: Optional[Dict[str, Any]] = None,
        default_callback_url: Optional[str] = None,
        default_callback_auth: Optional[CallbackAuth],
    ) -> str:
        """
        Updates the underlying infrastructure for a Model Endpoint.

        Args:
            billing_tags: Arbitrary tags passed to billing
            model_endpoint_record: The associated record of a model endpoint.
            min_workers: The minimum number of workers for the model endpoint.
            max_workers: The maximum number of workers for the model endpoint.
            per_worker: Autoscaling parameter used to set the target number of enqueued items per worker.
            concurrent_requests: The max number of concurrent requests for the worker to work on.
            cpus: The amount of CPU to use per worker for the model endpoint.
            gpus: The amount of GPU to use per worker for the model endpoint.
            memory: The amount of memory to use per worker for the model endpoint.
            gpu_type: The type of GPU to use per worker for the model endpoint.
            storage: The amount of storage to request per worker for the model endpoint.
            optimize_costs: Whether to automatically infer CPU and memory.
            child_fn_info: For pipelines.
            post_inference_hooks: A list of optional post-inference hooks to perform.
            labels: Labels to attach to the infrastructure for tracking purposes.
            prewarm: For async endpoints only, whether to have pods load resources on worker startup
                as opposed to on first request. If loading takes more than 5 minutes then set this
                to False
            high_priority: Makes all pods for this endpoint higher priority to enable faster pod spinup
                time. Higher priority pods will displace the lower priority dummy pods from shared pool.
            default_callback_url: The default callback URL to use for the model endpoint.
            default_callback_auth: The default callback auth to use for the model endpoint.

        Returns:
            A unique ID for the task to update the infrastructure resources.
        """

    @abstractmethod
    async def get_model_endpoint_infra(
        self, model_endpoint_record: ModelEndpointRecord
    ) -> Optional[ModelEndpointInfraState]:
        """
        Retrieves the model endpoint infrastructure state, given the deployment name.

        TODO: it would be better to retrieve endpoints directly using their `model_endpoint_id`.
        That's currently not possible because deployment names are created by a combination of
        user_id and endpoint_name, and they are also tagged accordingly. Instead, we probably
        want to tag deployments and their resources with `model_endpoint_id`.

        Args:
            model_endpoint_record: The associated record of a model endpoint.

        Returns:
            A domain entity containing the Model Endpoint infrastructure state. None if nonexistent.
        """

    @abstractmethod
    async def delete_model_endpoint_infra(self, model_endpoint_record: ModelEndpointRecord) -> bool:
        """
        Deletes the model endpoint infrastructure for a given deployment_name.

        Args:
            model_endpoint_record: The associated record of a model endpoint.

        Returns:
            Whether the model endpoint infrastructure was successfully deleted.
        """

    @abstractmethod
    async def restart_model_endpoint_infra(
        self, model_endpoint_record: ModelEndpointRecord
    ) -> None:
        """
        Restarts the model endpoint deployment.
        """
