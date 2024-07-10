import json
from enum import Enum
from typing import Any, Callable, Dict, Sequence, Set, Type, Union

from fastapi import routing
from fastapi._compat import GenerateJsonSchema, get_model_definitions
from fastapi.openapi.constants import REF_TEMPLATE
from fastapi.openapi.utils import get_openapi_path
from model_engine_server.common.dtos.tasks import (
    EndpointPredictV1Request,
    GetAsyncTaskV1Response,
    RequestSchema,
    ResponseSchema,
    SyncEndpointPredictV1Response,
    TaskStatus,
)
from model_engine_server.core.config import infra_config
from model_engine_server.domain.entities import (
    CallbackAuth,
    CallbackBasicAuth,
    CallbackmTLSAuth,
    ModelEndpointRecord,
    ModelEndpointsSchema,
    ModelEndpointType,
)
from model_engine_server.domain.gateways import ModelEndpointsSchemaGateway
from model_engine_server.infra.gateways.filesystem_gateway import FilesystemGateway
from pydantic import BaseModel
from starlette.routing import BaseRoute

# Caches the default model definition so we don't need to recompute every time
_default_model_definitions = None

API_REFERENCE_TITLE = "Launch Endpoints API Reference"
API_REFERENCE_VERSION = "1.0.0"


def predict_stub_async(payload: EndpointPredictV1Request) -> GetAsyncTaskV1Response:
    raise NotImplementedError


def predict_stub_sync(payload: EndpointPredictV1Request) -> SyncEndpointPredictV1Response:
    raise NotImplementedError


class LiveModelEndpointsSchemaGateway(ModelEndpointsSchemaGateway):
    """Gateway for the OpenAPI schema for live model endpoints backed by OpenAPI and FastAPI."""

    def __init__(self, filesystem_gateway: FilesystemGateway):
        self.filesystem_gateway = filesystem_gateway

    def get_model_endpoints_schema(
        self,
        model_endpoint_records: Sequence[ModelEndpointRecord],
    ) -> ModelEndpointsSchema:
        routes = []
        model_endpoint_names = []
        model_definitions = {}
        for record in model_endpoint_records:
            response_model: Type[BaseModel] = GetAsyncTaskV1Response
            predict_stub: Callable[[EndpointPredictV1Request], Any] = predict_stub_async
            base_route = "/v1/async-tasks"
            if record.endpoint_type == ModelEndpointType.SYNC:
                response_model = SyncEndpointPredictV1Response
                predict_stub = predict_stub_sync
                base_route = "/v1/sync-tasks"
            route = routing.APIRoute(
                f"{base_route}?model_endpoint_id={record.id}",
                predict_stub,
                response_model=response_model,
                name=record.name,
                methods=["POST"],
            )
            routes.append(route)
            definitions = self.get_schemas_from_model_endpoint_record(record)
            definitions = LiveModelEndpointsSchemaGateway.update_model_definitions_with_prefix(
                prefix=record.name, model_definitions=definitions
            )
            model_definitions.update(definitions)
            model_endpoint_names.append(record.name)

        schema = LiveModelEndpointsSchemaGateway.get_openapi(
            title=API_REFERENCE_TITLE,
            version=API_REFERENCE_VERSION,
            routes=routes,
            model_endpoint_names=model_endpoint_names,
            model_definitions=model_definitions,
        )
        return schema

    @staticmethod
    def get_openapi(
        *,
        title: str,
        version: str,
        openapi_version: str = "3.0.2",
        routes: Sequence[BaseRoute],
        model_endpoint_names: Sequence[str],
        model_definitions: Dict[str, Any],
    ) -> ModelEndpointsSchema:
        """Generate an OpenAPI schema for the given API routes.

        Args:
            title (str): The title of the API.
            version (str): The version of the API.
            openapi_version (str, optional): The OpenAPI version. Defaults to "3.0.2".
            routes (Sequence[BaseRoute]): The API routes.
            model_endpoint_names (Sequence[str]): The names of the model endpoints.
            model_definitions (Dict[str, Any]): The model definitions.

        Returns:
            ModelEndpointsSchema: The OpenAPI schema.
        """
        info: Dict[str, Any] = {"title": title, "version": version}
        output: Dict[str, Any] = {"openapi": openapi_version, "info": info}
        components: Dict[str, Dict[str, Any]] = {}
        paths: Dict[str, Dict[str, Any]] = {}
        operation_ids: Set[str] = set()
        for route, model_endpoint_name in zip(routes, model_endpoint_names):
            if isinstance(route, routing.APIRoute):
                prefix = model_endpoint_name
                model_name_map = LiveModelEndpointsSchemaGateway.get_model_name_map(prefix)
                schema_generator = GenerateJsonSchema(ref_template=REF_TEMPLATE)
                result = get_openapi_path(
                    route=route,
                    model_name_map=model_name_map,
                    operation_ids=operation_ids,
                    schema_generator=schema_generator,
                    field_mapping={},
                )
                if result:
                    path, security_schemes, path_definitions = result
                    if path:
                        paths.setdefault(route.path_format, {}).update(path)
                    if security_schemes:
                        components.setdefault("securitySchemes", {}).update(security_schemes)
                    if path_definitions:
                        model_definitions.update(path_definitions)
        if model_definitions:
            components["schemas"] = {k: model_definitions[k] for k in sorted(model_definitions)}
        if components:
            output["components"] = components
        output["paths"] = paths
        return ModelEndpointsSchema(**output)

    @staticmethod
    def update_model_definitions_with_prefix(
        *, prefix: str, model_definitions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update the model definitions for the given path.

        Args:
            prefix (str): The prefix.
            model_definitions (Dict[str, Any]): The model definitions.

        Returns:
            Dict[str, Any]: The updated model definitions.
        """
        models: Set[Union[Type[BaseModel], Type[Enum]]] = {
            CallbackAuth,
            CallbackBasicAuth,
            CallbackmTLSAuth,
            TaskStatus,
            EndpointPredictV1Request,
            GetAsyncTaskV1Response,
            SyncEndpointPredictV1Response,
        }
        definitions = get_model_definitions(
            flat_models=models,
            model_name_map=LiveModelEndpointsSchemaGateway.get_model_name_map(prefix),
        )
        user_definitions = {}
        for k, v in model_definitions.items():
            LiveModelEndpointsSchemaGateway.update_schema_refs_with_prefix(v, prefix)
            user_definitions[f"{prefix}-{k}"] = v
        definitions.update(user_definitions)
        return definitions

    @staticmethod
    def update_schema_refs_with_prefix(schema: Dict[str, Any], prefix: str) -> None:
        """Recursively update the schema references with the prefix."""
        if "$ref" in schema:
            schema["$ref"] = schema["$ref"].replace(
                "#/components/schemas/", f"#/components/schemas/{prefix}-"
            )
        for v in schema.values():
            if isinstance(v, dict):
                LiveModelEndpointsSchemaGateway.update_schema_refs_with_prefix(v, prefix)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        LiveModelEndpointsSchemaGateway.update_schema_refs_with_prefix(item, prefix)

    @staticmethod
    def get_model_name_map(prefix: str) -> Dict[Union[Type[BaseModel], Type[Enum]], str]:
        return {
            CallbackAuth: "CallbackAuth",
            CallbackBasicAuth: "CallbackBasicAuth",
            CallbackmTLSAuth: "CallbackmTLSAuth",
            TaskStatus: "TaskStatus",
            EndpointPredictV1Request: f"{prefix}-EndpointPredictRequest",
            GetAsyncTaskV1Response: f"{prefix}-GetAsyncTaskResponse",
            SyncEndpointPredictV1Response: f"{prefix}-SyncEndpointPredictResponse",
            RequestSchema: f"{prefix}-RequestSchema",
            ResponseSchema: f"{prefix}-ResponseSchema",
        }

    def get_schemas_from_model_endpoint_record(
        self,
        model_endpoint_record: ModelEndpointRecord,
    ) -> Dict[str, Any]:
        """Get the schemas from the model endpoint records.

        Args:
            model_endpoint_record: The model endpoint record.

        Returns:
            Dict[str, Any]: The schemas.
        """
        schema_location = model_endpoint_record.current_model_bundle.schema_location
        schema = None
        try:
            if schema_location is not None:
                with self.filesystem_gateway.open(
                    schema_location, "rb", aws_profile=infra_config().profile_ml_worker
                ) as f:
                    schema = json.load(f)
        finally:
            if schema is None:
                return LiveModelEndpointsSchemaGateway.get_default_model_definitions()

        return schema

    @staticmethod
    def get_default_model_definitions() -> Dict[str, Any]:
        global _default_model_definitions

        if _default_model_definitions is None:
            _default_model_definitions = get_model_definitions(
                flat_models={RequestSchema, ResponseSchema},
                model_name_map={
                    RequestSchema: "RequestSchema",
                    ResponseSchema: "ResponseSchema",
                },
            )

        return _default_model_definitions
