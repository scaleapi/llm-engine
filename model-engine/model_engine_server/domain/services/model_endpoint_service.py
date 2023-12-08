# Represents high-level CRUD operations for model endpoints.
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from model_engine_server.common.dtos.model_endpoints import ModelEndpointOrderBy
from model_engine_server.domain.entities import (
    CallbackAuth,
    CpuSpecificationType,
    GpuType,
    ModelEndpoint,
    ModelEndpointRecord,
    ModelEndpointsSchema,
    ModelEndpointType,
    ShadowModelEndpointRecord,
    StorageSpecificationType,
)
from model_engine_server.domain.gateways import (
    AsyncModelEndpointInferenceGateway,
    StreamingModelEndpointInferenceGateway,
    SyncModelEndpointInferenceGateway,
)
from model_engine_server.domain.gateways.inference_autoscaling_metrics_gateway import (
    InferenceAutoscalingMetricsGateway,
)


class ModelEndpointService(ABC):
    """
    Base class for Model Endpoint services.
    """

    @abstractmethod
    def get_async_model_endpoint_inference_gateway(
        self,
    ) -> AsyncModelEndpointInferenceGateway:
        """
        Returns the async model endpoint inference gateway.
        """

    @abstractmethod
    def get_sync_model_endpoint_inference_gateway(
        self,
    ) -> SyncModelEndpointInferenceGateway:
        """
        Returns the sync model endpoint inference gateway.
        """

    @abstractmethod
    def get_streaming_model_endpoint_inference_gateway(
        self,
    ) -> StreamingModelEndpointInferenceGateway:
        """
        Returns the sync model endpoint inference gateway.
        """

    @abstractmethod
    def get_inference_auto_scaling_metrics_gateway(
        self,
    ) -> InferenceAutoscalingMetricsGateway:
        """
        Returns the inference autoscaling metrics gateway.
        """

    @abstractmethod
    async def create_model_endpoint(
        self,
        *,
        name: str,
        created_by: str,
        model_bundle_id: str,
        endpoint_type: ModelEndpointType,
        metadata: Dict[str, Any],
        post_inference_hooks: Optional[List[str]],
        child_fn_info: Optional[Dict[str, Any]],
        cpus: CpuSpecificationType,
        gpus: int,
        memory: StorageSpecificationType,
        gpu_type: Optional[GpuType],
        storage: Optional[StorageSpecificationType],
        optimize_costs: bool,
        min_workers: int,
        max_workers: int,
        per_worker: int,
        labels: Dict[str, str],
        aws_role: str,
        results_s3_bucket: str,
        prewarm: bool,
        high_priority: Optional[bool],
        billing_tags: Optional[Dict[str, Any]] = None,
        owner: str,
        default_callback_url: Optional[str],
        default_callback_auth: Optional[CallbackAuth],
        public_inference: Optional[bool] = False,
        shadow_endpoints: Optional[List[ShadowModelEndpointRecord]] = None,
    ) -> ModelEndpointRecord:
        """
        Creates a model endpoint.
        Args:
            name: The name of the model endpoint.
            created_by: The user ID of the creator of the model endpoint.
            model_bundle_id: The unique ID of the model bundle to use for the model endpoint.
            endpoint_type: The type of the endpoint.
            metadata: Key-value metadata to attach to the model endpoint.
            post_inference_hooks: Optional hooks to perform after inference is comoplete.
            child_fn_info: For pipelines.
            cpus: The amount of CPUs to use per worker for the endpoint.
            gpus: The amount of GPUs to use per worker for the endpoint.
            memory: The amount of memory to use per worker for the endpoint.
            gpu_type: The type of GPU to use per worker for the endpoint.
            storage: The amount of storage to use per worker for the endpoint.
            optimize_costs: Whether to automatically infer CPU and memory.
            min_workers: The minimum number of workers for the endpoint.
            max_workers: The maximum number of workers for the endpoint.
            per_worker: The number of concurrent tasks to process per worker.
            labels: The labels to attach to the model endpoint.
            aws_role: The AWS role to use for the endpoint.
            results_s3_bucket: The S3 bucket to use for results of the endpoint.
            prewarm: For async endpoints only, whether to have pods load resources on worker startup
                as opposed to on first request. If loading takes more than 5 minutes then set this
                to False
            high_priority: Makes all pods for this endpoint higher priority to enable faster pod spinup
                time. Higher priority pods will displace the lower priority dummy pods from shared pool.
            billing_tags: Tags that get passed to scale's billing infra
            owner: The team ID of the creator of the model endpoint.
            default_callback_url: The default callback URL to use for the model endpoint.
            default_callback_auth: The default callback auth to use for the model endpoint.
            public_inference: Whether to allow public inference.
            shadow_endpoints: The shadow endpoints to deploy with the model endpoint.
        Returns:
            A Model Endpoint Record domain entity object of the created endpoint.
        Raises:
            ObjectAlreadyExistsException: if a model endpoint with the given name already exists.
        """

    @abstractmethod
    async def delete_model_endpoint(self, model_endpoint_id: str) -> None:
        """
        Deletes the endpoint.
        Args:
            model_endpoint_id: The unique ID of the model endpoint to delete.
        Returns:
            None.
        Raises:
            EndpointDeleteFailedException: if the server couldn't delete resources for some reason
                (corresponds to an HTTP 500)
            ObjectNotFoundException: if the endpoint does not exist
                (corresponds to an HTTP 404)
            ExistingEndpointOperationInProgressException: if the endpoint is currently being edited
                (corresponds to an HTTP 409)
        """

    @abstractmethod
    async def get_model_endpoint(self, model_endpoint_id: str) -> Optional[ModelEndpoint]:
        """
        Gets a model endpoint.
        Args:
            model_endpoint_id: The ID of the model endpoint.
        Returns:
            A Model Endpoint domain entity, or None if not found.
        """

    @abstractmethod
    async def get_model_endpoints_schema(self, owner: str) -> ModelEndpointsSchema:
        """
        Gets the schema for model endpoints.
        Args:
            owner: The owner of the model endpoints.
        Returns:
            A Model Endpoint schema entity.
        """

    @abstractmethod
    async def get_model_endpoint_record(
        self, model_endpoint_id: str
    ) -> Optional[ModelEndpointRecord]:
        """
        Gets the record of a model endpoint.
        Args:
            model_endpoint_id: The ID of the model endpoint.
        Returns:
            A Model Endpoint Record domain entity, or None if not found.
        """

    @abstractmethod
    async def list_model_endpoints(
        self,
        owner: Optional[str],
        name: Optional[str],
        order_by: Optional[ModelEndpointOrderBy],
    ) -> List[ModelEndpoint]:
        """
        Lists model endpoints.
        Args:
            owner: The user ID of the owner of the endpoints.
            name: An optional name of the endpoint used for filtering endpoints.
            order_by: The ordering to output the Model Endpoints.
        Returns:
            A Model Endpoint Record domain entity, or None if not found.
        """

    @abstractmethod
    async def update_model_endpoint(
        self,
        *,
        model_endpoint_id: str,
        model_bundle_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,  # TODO: JSON type
        post_inference_hooks: Optional[Any] = None,
        cpus: Optional[CpuSpecificationType] = None,
        gpus: Optional[int] = None,
        memory: Optional[StorageSpecificationType] = None,
        gpu_type: Optional[GpuType] = None,
        storage: Optional[StorageSpecificationType] = None,
        optimize_costs: Optional[bool] = None,
        min_workers: Optional[int] = None,
        max_workers: Optional[int] = None,
        per_worker: Optional[int] = None,
        labels: Optional[Dict[str, str]] = None,
        prewarm: Optional[bool] = None,
        high_priority: Optional[bool] = None,
        billing_tags: Optional[Dict[str, Any]] = None,
        default_callback_url: Optional[str] = None,
        default_callback_auth: Optional[CallbackAuth] = None,
        public_inference: Optional[bool] = None,
        shadow_endpoints: Optional[List[ShadowModelEndpointRecord]] = None,
    ) -> ModelEndpointRecord:
        """
        Updates a model endpoint.

        Args:
            model_endpoint_id: The unique ID of the model endpoint to update.
            model_bundle_id: The unique ID of the model bundle to use for the model endpoint.
            metadata: Key-value metadata to attach to the model endpoint.
            post_inference_hooks: Optional hooks to perform after inference is comoplete.
            cpus: The amount of CPUs to use per worker for the endpoint.
            gpus: The amount of GPUs to use per worker for the endpoint.
            memory: The amount of memory to use per worker for the endpoint.
            gpu_type: The type of GPU to use per worker for the endpoint.
            storage: The amount of storage to use per worker for the endpoint.
            optimize_costs: Whether to automatically infer CPU and memory.
            min_workers: The minimum number of workers for the endpoint.
            max_workers: The maximum number of workers for the endpoint.
            per_worker: The number of concurrent tasks to process per worker.
            labels: The labels to attach to the model endpoint.
            prewarm: For async endpoints only, whether to have pods load resources on worker startup
                as opposed to on first request. If loading takes more than 5 minutes then set this
                to False
            high_priority: Makes all pods for this endpoint higher priority to enable faster pod spinup
                time. Higher priority pods will displace the lower priority dummy pods from shared pool.
            billing_tags: Tags that get passed to scale's billing infra
            default_callback_url: The default callback URL to use for the model endpoint.
            default_callback_auth: The default callback auth to use for the model endpoint.
            public_inference: Whether to allow public inference.
            shadow_endpoints: The shadow endpoints to deploy with the model endpoint.

        Returns:
            A Model Endpoint Record domain entity object of the updated endpoint.
        Raises:
            ObjectNotFoundException: if the endpoint does not exist
                (corresponds to an HTTP 404)
            ExistingEndpointOperationInProgressException: if the endpoint is currently being edited
                (corresponds to an HTTP 409)
        """
