from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from llm_engine_server.common.dtos.model_bundles import ModelBundleOrderBy
from llm_engine_server.domain.entities import (
    ModelBundle,
    ModelBundleFlavors,
    ModelBundlePackagingType,
)


class ModelBundleRepository(ABC):
    """
    Base class for ModelBundle repositories.
    """

    @abstractmethod
    async def create_model_bundle(
        self,
        *,
        name: str,
        created_by: str,
        owner: str,
        model_artifact_ids: List[str],
        schema_location: Optional[str],
        metadata: Dict[str, Any],
        flavor: ModelBundleFlavors,
        # LEGACY FIELDS
        location: str,
        requirements: List[str],
        env_params: Dict[str, Any],
        packaging_type: ModelBundlePackagingType,
        app_config: Optional[Dict[str, Any]],
    ) -> ModelBundle:
        """
        Creates a Model Bundle.

        Args:
            name: The name of the Model Bundle.
            created_by: The user creating the Model Bundle.
            owner: Team who owns the Model Bundle
            model_artifact_ids: The IDs of the model artifact(s) associated with the Model Bundle.
            schema_location: The URL of the schema file.
            metadata: Key-value metadata associated with the Model Bundle.
            flavor: The flavor of the Model Bundle.
            # LEGACY FIELDS
            location: The URL of the model bundle file(s).
            requirements: A list of requirements of the form `numpy==0.4.0`.
            env_params: Key-value pairs specifying environment variables.
            packaging_type: The method used to package the code (e.g. cloudpickle or zip).
            app_config: JSON configuration to initialize the app.

        Returns:
            A Model Bundle domain entity.
        """

    @abstractmethod
    async def list_model_bundles(
        self, owner: str, name: Optional[str], order_by: Optional[ModelBundleOrderBy]
    ) -> Sequence[ModelBundle]:
        """
        Lists the Model Bundles associated with a given owner and name.

        Args:
            owner: The owner of the model bundle(s).
            name: The name of the model bundle(s).
            order_by: The ordering (newest or oldest) to output the Model Bundle versions.

        Returns:
            A sequence of Model Bundle domain entities.
        """

    @abstractmethod
    async def get_model_bundle(self, model_bundle_id: str) -> Optional[ModelBundle]:
        """
        Retrieves a single Model Bundle by ID

        Args:
            model_bundle_id: The ID of the model bundle.

        Returns:
            The associated Model Bundle, or None if it could not be found.
        """

    @abstractmethod
    async def get_latest_model_bundle_by_name(self, owner: str, name: str) -> Optional[ModelBundle]:
        """
        Retrieves the latest Model Bundle by owner and name.

        Args:
            owner: the name of the owner.
            name: the name of the bundle

        Returns:
            The associated Model Bundle, or None if not found.
        """
