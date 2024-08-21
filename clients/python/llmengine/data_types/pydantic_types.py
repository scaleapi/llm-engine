from pydantic import BaseModel as PydanticBaseModel
from pydantic import (  # noqa: F401
    ConfigDict,
    Field,
    HttpUrl,
    RootModel,
    ValidationError,
    model_validator,
)


class BaseModel(PydanticBaseModel):
    """Common pydantic configurations for model engine"""

    model_config = ConfigDict(protected_namespaces=())
