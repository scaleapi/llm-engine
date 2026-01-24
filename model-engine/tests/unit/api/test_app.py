from fastapi.testclient import TestClient
from model_engine_server.api.app import OPENAPI_SCHEMA_RENAME_PATTERNS, _rename_openapi_schemas


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
