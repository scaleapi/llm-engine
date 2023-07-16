from abc import ABC, abstractmethod
from typing import Optional

from llm_engine_server.domain.entities.model_bundle_entity import ModelBundleFrameworkType


class ModelPrimitiveGateway(ABC):
    """
    Base class for interactions with Scale Model Primitive.
    """

    @abstractmethod
    async def create_model_artifact(
        self,
        model_artifact_name: str,
        location: str,
        framework_type: ModelBundleFrameworkType,
        created_by: str,
    ) -> Optional[str]:
        """
        Creates the Model Artifact.

        Args:
            model_artifact_name: The name of the artifact to create.
            location: The location of the model artifact file.
            framework_type: The type of framework used for the model artifact (e.g. tensorflow)
            created_by: The ID of the user who is creating the model artifact.

        Returns:
            The ID of the created Model Artifact, or None if it failed to create.
        """
