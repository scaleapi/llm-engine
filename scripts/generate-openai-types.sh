#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=${SCRIPT_DIR}/..

DEST_DIR=${BASE_DIR}/model-engine/model_engine_server/common/types/gen
OPENAI_SPEC=${SCRIPT_DIR}/openai-spec.yaml

# Generate OpenAPI types for server
datamodel-codegen \
    --input ${OPENAI_SPEC} \
    --input-file-type openapi \
    --output ${DEST_DIR}/openai.py \
    --output-model-type pydantic_v2.BaseModel \
    --enum-field-as-literal all \
    --field-constraints \
    --use-annotated

CLIENT_DIR=${BASE_DIR}/clients/python/llmengine/data_types/gen

# Generate OpenAPI types for server
datamodel-codegen \
    --input ${OPENAI_SPEC} \
    --input-file-type openapi \
    --output ${CLIENT_DIR}/openai.py \
    --output-model-type pydantic.BaseModel \
    --enum-field-as-literal all \
    --field-constraints \
    --use-annotated

sed -i '1s/^/# mypy: ignore-errors\n/' ${CLIENT_DIR}/openai.py