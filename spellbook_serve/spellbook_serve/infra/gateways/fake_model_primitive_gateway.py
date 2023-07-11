from typing import Optional

from spellbook_serve.domain.entities import ModelBundleFrameworkType
from spellbook_serve.domain.gateways import ModelPrimitiveGateway


class FakeModelPrimitiveGateway(ModelPrimitiveGateway):
    def __init__(self):
        self.db = {}

    async def create_model_artifact(
        self,
        model_artifact_name: str,
        location: str,
        framework_type: ModelBundleFrameworkType,
        created_by: str,
    ) -> Optional[str]:
        if model_artifact_name in self.db:
            return None
        self.db[model_artifact_name] = dict(
            model_artifact_name=model_artifact_name,
            location=location,
            framework_type=framework_type,
            created_by=created_by,
        )
        return f"{created_by}-{model_artifact_name}"
