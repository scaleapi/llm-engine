import pydantic

PYDANTIC_V2 = hasattr(pydantic, "VERSION") and pydantic.VERSION.startswith("2.")

if PYDANTIC_V2:
    from pydantic.v1 import BaseModel, Field, HttpUrl  # noqa: F401

else:
    from pydantic import BaseModel, Field, HttpUrl  # type: ignore # noqa: F401
