from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, RootModel, ValidationError, model_validator  # noqa: F401


class BaseModel(PydanticBaseModel):
    """Common pydantic configurations for model engine"""

    model_config = ConfigDict(protected_namespaces=())
