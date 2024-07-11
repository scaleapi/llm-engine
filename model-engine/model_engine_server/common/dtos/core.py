from typing_extensions import Annotated
from pydantic import BaseModel, BeforeValidator, ConfigDict, HttpUrl, TypeAdapter

# See: https://github.com/pydantic/pydantic/issues/7186
# pydantic v2 doesn't treat HttpUrl the same way as in v1 which causes various issue
# This is an attempt to make it behave as similar as possible
HttpUrlTypeAdapter = TypeAdapter(HttpUrl)
HttpUrlStr = Annotated[
    str,
    BeforeValidator(lambda value: HttpUrlTypeAdapter.validate_python(value) and value),
]

class LLMEngineModel(BaseModel):
    """Common pydantic configurations for model engine"""
    model_config = ConfigDict(protected_namespaces=())
