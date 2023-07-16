import pytest
from llm_engine_server.domain.entities import ModelEndpoint
from llm_engine_server.infra.gateways.live_model_endpoints_schema_gateway import (
    LiveModelEndpointsSchemaGateway,
)


@pytest.fixture
def live_model_endpoints_schema_gateway(
    fake_filesystem_gateway,
) -> LiveModelEndpointsSchemaGateway:
    return LiveModelEndpointsSchemaGateway(filesystem_gateway=fake_filesystem_gateway)


@pytest.mark.parametrize(
    "model_definitions,expected",
    [
        (
            {
                "RequestSchema": {
                    "title": "UserRequestSchema",
                    "type": "object",
                    "properties": {
                        "query": {"title": "Query", "type": "string"},
                        "idx": {"title": "Idx", "type": "integer"},
                    },
                    "required": ["query", "idx"],
                },
                "UserResponseSubSchema": {
                    "title": "UserResponseSubSchema",
                    "type": "object",
                    "properties": {
                        "prediction": {"title": "Prediction", "type": "string"},
                        "confidence": {"title": "Confidence", "type": "number"},
                    },
                    "required": ["prediction", "confidence"],
                },
                "ResponseSchema": {
                    "title": "UserResponseSchema",
                    "type": "object",
                    "properties": {
                        "prediction_list": {
                            "title": "Prediction List",
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/UserResponseSubSchema"},
                        },
                        "prediction": {"$ref": "#/components/schemas/UserResponseSubSchema"},
                        "prediction_dict": {
                            "title": "Prediction Dict",
                            "type": "object",
                            "additionalProperties": {
                                "$ref": "#/components/schemas/UserResponseSubSchema"
                            },
                        },
                    },
                    "required": ["prediction_list", "prediction", "prediction_dict"],
                },
            },
            {
                "RequestSchema": {
                    "title": "UserRequestSchema",
                    "type": "object",
                    "properties": {
                        "query": {"title": "Query", "type": "string"},
                        "idx": {"title": "Idx", "type": "integer"},
                    },
                    "required": ["query", "idx"],
                },
                "UserResponseSubSchema": {
                    "title": "UserResponseSubSchema",
                    "type": "object",
                    "properties": {
                        "prediction": {"title": "Prediction", "type": "string"},
                        "confidence": {"title": "Confidence", "type": "number"},
                    },
                    "required": ["prediction", "confidence"],
                },
                "ResponseSchema": {
                    "title": "UserResponseSchema",
                    "type": "object",
                    "properties": {
                        "prediction_list": {
                            "title": "Prediction List",
                            "type": "array",
                            "items": {
                                "$ref": "#/components/schemas/test_prefix-UserResponseSubSchema"
                            },
                        },
                        "prediction": {
                            "$ref": "#/components/schemas/test_prefix-UserResponseSubSchema"
                        },
                        "prediction_dict": {
                            "title": "Prediction Dict",
                            "type": "object",
                            "additionalProperties": {
                                "$ref": "#/components/schemas/test_prefix-UserResponseSubSchema"
                            },
                        },
                    },
                    "required": ["prediction_list", "prediction", "prediction_dict"],
                },
            },
        ),
        (
            {
                "MyEnum": {
                    "description": "An enumeration.",
                    "enum": ["one", "two"],
                    "title": "MyEnum",
                    "type": "string",
                },
                "RequestSchema": {
                    "properties": {"value": {"$ref": "#/components/schemas/MyEnum"}},
                    "required": ["value"],
                    "title": "UserRequestSchema",
                    "type": "object",
                },
                "ResponseSchema": {
                    "properties": {
                        "result": {
                            "additionalProperties": {
                                "items": {"$ref": "#/components/schemas/MyEnum"},
                                "type": "array",
                            },
                            "title": "Result",
                            "type": "object",
                        }
                    },
                    "required": ["result"],
                    "title": "UserResponseSchema",
                    "type": "object",
                },
            },
            {
                "MyEnum": {
                    "description": "An enumeration.",
                    "enum": ["one", "two"],
                    "title": "MyEnum",
                    "type": "string",
                },
                "RequestSchema": {
                    "properties": {"value": {"$ref": "#/components/schemas/test_prefix-MyEnum"}},
                    "required": ["value"],
                    "title": "UserRequestSchema",
                    "type": "object",
                },
                "ResponseSchema": {
                    "properties": {
                        "result": {
                            "additionalProperties": {
                                "items": {"$ref": "#/components/schemas/test_prefix-MyEnum"},
                                "type": "array",
                            },
                            "title": "Result",
                            "type": "object",
                        }
                    },
                    "required": ["result"],
                    "title": "UserResponseSchema",
                    "type": "object",
                },
            },
        ),
    ],
)
def test_update_schema_refs_with_prefix(
    live_model_endpoints_schema_gateway, model_definitions, expected
):
    live_model_endpoints_schema_gateway.update_schema_refs_with_prefix(
        schema=model_definitions,
        prefix="test_prefix",
    )
    assert model_definitions == expected


def test_get_model_endpoints_schema_success(
    live_model_endpoints_schema_gateway: LiveModelEndpointsSchemaGateway,
    model_endpoint_1: ModelEndpoint,
    model_endpoint_2: ModelEndpoint,
):
    schema = live_model_endpoints_schema_gateway.get_model_endpoints_schema(
        model_endpoint_records=[model_endpoint_1.record, model_endpoint_2.record],
    )
    assert schema is not None
