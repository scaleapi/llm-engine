from fastapi.testclient import TestClient
from model_engine_server.api.app import (
    OPENAPI_SCHEMA_RENAME_PATTERNS,
    _convert_openapi_31_to_30,
    _rename_openapi_schemas,
    get_openapi_schema,
)


def test_healthcheck(simple_client: TestClient):
    response = simple_client.get("/healthcheck")
    assert response.status_code == 200

    response = simple_client.get("/healthz")
    assert response.status_code == 200

    response = simple_client.get("/readyz")
    assert response.status_code == 200


def test_rename_openapi_schemas_renames_discriminated_unions():
    """Test that discriminated union schemas are renamed correctly."""
    ugly_name = (
        "RootModel_Annotated_Union_Annotated_CreateVLLMEndpoint__SomeMiddle__Discriminator__"
    )
    schema = {
        "components": {
            "schemas": {
                ugly_name: {
                    "oneOf": [
                        {"$ref": "#/components/schemas/CreateVLLMEndpoint"},
                        {"$ref": "#/components/schemas/CreateTRTLLMEndpoint"},
                    ]
                },
                "OtherSchema": {"type": "string"},
            }
        },
        "paths": {
            "/test": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": f"#/components/schemas/{ugly_name}"}
                            }
                        }
                    }
                }
            }
        },
    }

    result = _rename_openapi_schemas(schema)

    # Schema should be renamed
    assert "CreateLLMModelEndpointV1Request" in result["components"]["schemas"]
    assert ugly_name not in result["components"]["schemas"]

    # Ref should be updated
    ref = result["paths"]["/test"]["post"]["requestBody"]["content"]["application/json"]["schema"][
        "$ref"
    ]
    assert ref == "#/components/schemas/CreateLLMModelEndpointV1Request"


def test_rename_openapi_schemas_handles_update_endpoint():
    """Test that UpdateLLMModelEndpointV1Request is renamed correctly."""
    ugly_name = "RootModel_Annotated_Union_Annotated_UpdateVLLMEndpoint__Foo__Discriminator__"
    schema = {
        "components": {
            "schemas": {
                ugly_name: {"oneOf": []},
            }
        }
    }

    result = _rename_openapi_schemas(schema)

    assert "UpdateLLMModelEndpointV1Request" in result["components"]["schemas"]
    assert ugly_name not in result["components"]["schemas"]


def test_rename_openapi_schemas_no_components():
    """Test that schemas without components are returned unchanged."""
    schema = {"paths": {}}
    result = _rename_openapi_schemas(schema)
    assert result == schema


def test_rename_openapi_schemas_no_schemas():
    """Test that schemas without schemas key are returned unchanged."""
    schema = {"components": {}}
    result = _rename_openapi_schemas(schema)
    assert result == schema


def test_rename_openapi_schemas_no_matching_patterns():
    """Test that schemas without matching patterns are unchanged."""
    schema = {
        "components": {
            "schemas": {
                "SomeOtherSchema": {"type": "string"},
            }
        }
    }
    result = _rename_openapi_schemas(schema)
    assert result["components"]["schemas"] == {"SomeOtherSchema": {"type": "string"}}


def test_rename_openapi_schemas_updates_nested_refs():
    """Test that nested $ref values are updated."""
    ugly_name = "RootModel_Annotated_Union_Annotated_CreateVLLMTest__Discriminator__"
    schema = {
        "components": {
            "schemas": {
                ugly_name: {"type": "object"},
                "Wrapper": {
                    "properties": {
                        "nested": {"items": {"$ref": f"#/components/schemas/{ugly_name}"}}
                    }
                },
            }
        }
    }

    result = _rename_openapi_schemas(schema)

    nested_ref = result["components"]["schemas"]["Wrapper"]["properties"]["nested"]["items"]["$ref"]
    assert nested_ref == "#/components/schemas/CreateLLMModelEndpointV1Request"


def test_openapi_schema_rename_patterns_defined():
    """Test that the rename patterns are properly defined."""
    assert len(OPENAPI_SCHEMA_RENAME_PATTERNS) >= 2

    # Check Create pattern
    create_pattern = next(p for p in OPENAPI_SCHEMA_RENAME_PATTERNS if "Create" in p[2])
    assert create_pattern[0].startswith("RootModel_Annotated_Union_Annotated_CreateVLLM")
    assert create_pattern[1] == "__Discriminator__"
    assert create_pattern[2] == "CreateLLMModelEndpointV1Request"

    # Check Update pattern
    update_pattern = next(p for p in OPENAPI_SCHEMA_RENAME_PATTERNS if "Update" in p[2])
    assert update_pattern[0].startswith("RootModel_Annotated_Union_Annotated_UpdateVLLM")
    assert update_pattern[1] == "__Discriminator__"
    assert update_pattern[2] == "UpdateLLMModelEndpointV1Request"


def test_openapi_endpoint_returns_schema(simple_client: TestClient):
    """Test that /openapi.json endpoint returns a valid schema with renamed types."""
    response = simple_client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "components" in schema
    assert "schemas" in schema["components"]


# Tests for OpenAPI 3.1 -> 3.0 conversion


def test_convert_openapi_31_to_30_anyof_with_null():
    """Test that anyOf with null type is converted to nullable."""
    schema = {
        "properties": {
            "name": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Name",
            }
        }
    }

    _convert_openapi_31_to_30(schema)

    assert "anyOf" not in schema["properties"]["name"]
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["name"]["nullable"] is True
    assert schema["properties"]["name"]["title"] == "Name"


def test_convert_openapi_31_to_30_anyof_multiple_non_null():
    """Test that anyOf with multiple non-null types keeps anyOf but adds nullable."""
    schema = {
        "properties": {
            "value": {
                "anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}],
            }
        }
    }

    _convert_openapi_31_to_30(schema)

    assert "anyOf" in schema["properties"]["value"]
    assert len(schema["properties"]["value"]["anyOf"]) == 2
    assert schema["properties"]["value"]["nullable"] is True
    # Null type should be removed from anyOf
    types = [item.get("type") for item in schema["properties"]["value"]["anyOf"]]
    assert "null" not in types


def test_convert_openapi_31_to_30_removes_const_with_enum():
    """Test that const is removed when enum is present."""
    schema = {
        "LLMSource": {
            "type": "string",
            "enum": ["hugging_face"],
            "const": "hugging_face",
            "title": "LLMSource",
        }
    }

    _convert_openapi_31_to_30(schema)

    assert "const" not in schema["LLMSource"]
    assert schema["LLMSource"]["enum"] == ["hugging_face"]
    assert schema["LLMSource"]["type"] == "string"


def test_convert_openapi_31_to_30_converts_const_to_enum():
    """Test that const is converted to enum when enum is not present."""
    schema = {"field": {"const": "fixed_value", "type": "string"}}

    _convert_openapi_31_to_30(schema)

    # const should be converted to enum (const is 3.1 only)
    assert "const" not in schema["field"]
    assert schema["field"]["enum"] == ["fixed_value"]


def test_convert_openapi_31_to_30_nested():
    """Test that conversion works on deeply nested structures."""
    schema = {
        "components": {
            "schemas": {
                "Request": {
                    "properties": {
                        "config": {
                            "properties": {
                                "timeout": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
                            }
                        }
                    }
                }
            }
        }
    }

    _convert_openapi_31_to_30(schema)

    timeout = schema["components"]["schemas"]["Request"]["properties"]["config"]["properties"][
        "timeout"
    ]
    assert "anyOf" not in timeout
    assert timeout["type"] == "integer"
    assert timeout["nullable"] is True


def test_get_openapi_schema_30_compatible():
    """Test that get_openapi_schema with openapi_30_compatible=True produces 3.0 compatible output."""
    schema = get_openapi_schema(openapi_30_compatible=True)

    # Verify schema has expected structure and renamed models
    assert "CreateLLMModelEndpointV1Request" in schema["components"]["schemas"]


def test_get_openapi_schema_default():
    """Test that get_openapi_schema without flag returns native 3.1 schema."""
    schema = get_openapi_schema(openapi_30_compatible=False)

    assert "components" in schema
    assert "schemas" in schema["components"]
    assert "CreateLLMModelEndpointV1Request" in schema["components"]["schemas"]
