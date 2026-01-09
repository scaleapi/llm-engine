from typing import Any, Type, TypeVar

from pydantic import AnyHttpUrl as PyAnyHttpUrl
from pydantic import AnyUrl as PyAnyUrl
from pydantic import AnyWebsocketUrl as PyAnyWebsocketUrl
from pydantic import BaseModel as PydanticBaseModel
from pydantic import model_validator  # noqa: F401
from pydantic import ConfigDict, Field  # noqa: F401
from pydantic import FileUrl as PyFileUrl
from pydantic import FtpUrl as PyFtpUrl
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler  # noqa: F401
from pydantic import HttpUrl as PyHttpUrl
from pydantic import RootModel, TypeAdapter, ValidationError  # noqa: F401
from pydantic import WebsocketUrl as PyWebsocketUrl
from pydantic import constr  # noqa: F401
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema


class BaseModel(PydanticBaseModel):
    """Common pydantic configurations for model engine"""

    model_config = ConfigDict(protected_namespaces=())


# See https://github.com/patrsc/pydantic-string-url
#  just copied it over cause it was a single file

"""Pydantic URL types based on strings."""


T = TypeVar("T", bound=PyAnyUrl)


class AnyUrl(str):
    """Pydantic's AnyUrl based on str."""

    _pydantic_type = PyAnyUrl
    _example_url = "http://www.example.com/"

    def __init__(self, url: str) -> None:
        """Initialize."""
        pydantic_url = validate_url(url, self._pydantic_type)
        super().__init__()
        self.url = pydantic_url

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,  # pylint: disable=unused-argument
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Get pydantic core schema."""
        return core_schema.no_info_after_validator_function(cls._validate, handler(str))

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Get pydantic JSON schema."""
        json_schema = handler(schema)
        json_schema = handler.resolve_ref_schema(json_schema)
        json_schema["format"] = "uri"
        json_schema["minLength"] = 1
        json_schema["maxLength"] = 65536
        json_schema["examples"] = [cls._example_url]
        return json_schema

    @classmethod
    def _validate(cls, __input_value: str) -> "AnyUrl":
        return cls(__input_value)


def validate_url(s: str, cls: Type[T]) -> T:
    """Validate if string has the format of a proper URL or given Pydantic type."""
    # This uses pydantic's class just for validation.
    a = TypeAdapter(cls)
    url = a.validate_python(s, strict=True)
    return url


class AnyHttpUrl(AnyUrl):
    """Pydantic's AnyHttpUrl based on str."""

    _pydantic_type = PyAnyHttpUrl
    _example_url = "http://www.example.com/"


class HttpUrl(AnyUrl):
    """Pydantic's HttpUrl based on str."""

    _pydantic_type = PyHttpUrl
    _example_url = "http://www.example.com/"


class AnyWebsocketUrl(AnyUrl):
    """Pydantic's AnyWebsocketUrl based on str."""

    _pydantic_type = PyAnyWebsocketUrl
    _example_url = "ws://www.example.com/"


class WebsocketUrl(AnyUrl):
    """Pydantic's WebsocketUrl based on str."""

    _pydantic_type = PyWebsocketUrl
    _example_url = "ws://www.example.com/"


class FileUrl(AnyUrl):
    """Pydantic's FileUrl based on str."""

    _pydantic_type = PyFileUrl
    _example_url = "file://www.example.com/"


class FtpUrl(AnyUrl):
    """Pydantic's FtpUrl based on str."""

    _pydantic_type = PyFtpUrl
    _example_url = "ftp://www.example.com/"
