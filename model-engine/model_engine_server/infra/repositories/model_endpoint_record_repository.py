from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from typing import Any, Dict, List, Optional, Sequence

from model_engine_server.common.dtos.model_endpoints import ModelEndpointOrderBy
from model_engine_server.domain.entities import ModelEndpointRecord

__all__: Sequence[str] = ("ModelEndpointRecordRepository",)


class ModelEndpointRecordRepository(ABC):
    """
    Base class for Model Endpoint Record repositories.
    """

    class LockContext(AbstractAsyncContextManager):
        """
        Base class for lock contexts for Model Endpoint
        """

        @abstractmethod
        def lock_acquired(self) -> bool:
            """
            Returns: whether the lock was acquired.
            """

    @abstractmethod
    def get_lock_context(self, model_endpoint_record: ModelEndpointRecord) -> LockContext:
        """
        Returns a Context Manager object for locking the model endpoint record.

        Args:
            model_endpoint_record: The model endpoint record to lock.

        Returns:
            Context manager object for the lock.
        """

    @abstractmethod
    async def create_model_endpoint_record(  # TODO probably don't need multinode
        self,
        *,
        name: str,
        created_by: str,
        model_bundle_id: str,
        metadata: Optional[Dict[str, Any]],
        endpoint_type: str,
        destination: str,
        creation_task_id: str,
        status: str,
        owner: str,
        public_inference: Optional[bool] = False,
    ) -> ModelEndpointRecord:
        """
        Creates an entry for endpoint tracking data, but not the actual compute resources.

        Args:
            name: Name of endpoint
            created_by: User who created endpoint
            model_bundle_id: Bundle the endpoint uses
            metadata: Arbitrary dictionary containing user-defined metadata
            endpoint_type: Type of endpoint i.e. async/sync
            destination: The queue name (async) or deployment name (sync) of the endpoint, used for routing requests
            creation_task_id: The celery task id corresponding to endpoint creation
            status: A status field on the endpoint, keeps track of endpoint state,
                used to coordinate edit operations on the endpoint
            owner: Team who owns endpoint
            public_inference: Whether the endpoint is publicly accessible

        Returns:
            A Model Endpoint Record domain entity.
        """

    @abstractmethod
    async def update_model_endpoint_record(
        self,
        *,
        model_endpoint_id: str,
        model_bundle_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        creation_task_id: Optional[str] = None,
        destination: Optional[str] = None,
        status: Optional[str] = None,
        public_inference: Optional[bool] = None,
    ) -> Optional[ModelEndpointRecord]:
        """
        Updates the entry for endpoint tracking data with the given new values. Only these values are editable.

        Args:
            model_endpoint_id: Unique ID for the model endpoint to update
            model_bundle_id: Unique ID for the model bundle the endpoint uses
            metadata: Arbitrary dictionary containing user-defined metadata
            creation_task_id: The task id corresponding to endpoint creation
            destination: The destination where async tasks should be sent.
            status: Status field on the endpoint, used to coordinate endpoint edit operations
            public_inference: Whether the endpoint is publicly accessible

        Returns:
            A Model Endpoint Record domain entity if found, else None.
        """

    @abstractmethod
    async def list_model_endpoint_records(
        self,
        owner: Optional[str],
        name: Optional[str],
        order_by: Optional[ModelEndpointOrderBy],
    ) -> List[ModelEndpointRecord]:
        """
        Lists all the records of model endpoints given the filters.

        Args:
            owner: The user ID of the creator of the endpoints.
            name: An optional name of the endpoint used for filtering endpoints.
            order_by: The ordering to output the Model Endpoints.

        Returns:
            A list of Model Endpoint Record domain entities.
        """

    @abstractmethod
    async def list_llm_model_endpoint_records(
        self,
        owner: Optional[str],
        name: Optional[str],
        order_by: Optional[ModelEndpointOrderBy],
    ) -> List[ModelEndpointRecord]:
        """
        Lists all the records of LLM model endpoints given the filters.

        Args:
            owner: The user ID of the creator of the endpoints.
            name: An optional name of the endpoint used for filtering endpoints.
            order_by: The ordering to output the Model Endpoints.

        Returns:
            A list of LLM Model Endpoint Record domain entities.
        """

    @abstractmethod
    async def get_model_endpoint_record(
        self, model_endpoint_id: str
    ) -> Optional[ModelEndpointRecord]:
        """
        Gets a model endpoint record.

        Args:
            model_endpoint_id: The unique ID of the Model Endpoint Record to get.

        Returns:
            A Model Endpoint Record domain entity if found, else None.
        """

    @abstractmethod
    async def get_llm_model_endpoint_record(
        self, model_endpoint_name: str
    ) -> Optional[ModelEndpointRecord]:
        """
        Gets an LLM model endpoint record.

        Args:
            model_endpoint_name: The unique name of the LLM Model Endpoint Record to get.

        Returns:
            A Model Endpoint Record domain entity if found, else None.
        """

    @abstractmethod
    async def delete_model_endpoint_record(self, model_endpoint_id: str) -> bool:
        """
        Deletes a model endpoint record.

        Args:
            model_endpoint_id: The unique ID of the Model Endpoint Record to delete.

        Returns:
            Whether a model endpoint was successfully deleted.
        """
