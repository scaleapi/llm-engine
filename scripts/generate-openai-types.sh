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
    --strict-nullable \
    --use-annotated

CLIENT_DIR=${BASE_DIR}/clients/python/llmengine/data_types/gen

# Generate OpenAPI types for client
#   Client is using pydantic v1
datamodel-codegen \
    --input ${OPENAI_SPEC} \
    --input-file-type openapi \
    --output ${CLIENT_DIR}/openai.py \
    --output-model-type pydantic.BaseModel \
    --enum-field-as-literal all \
    --field-constraints \
    --strict-nullable \
    --use-annotated

# Ignore mypy for this file
#   I tried updating mypy.ini to ignore this file, but it didn't work
sed -i '1s/^/# mypy: ignore-errors\n/' ${CLIENT_DIR}/openai.py

# Add conditional import for pydantic v1 and v2
# replace line starting with 'from pydantic <import stuff>' with the following multiline python code
# import pydantic
# PYDANTIC_V2 = hasattr(pydantic, "VERSION") and pydantic.VERSION.startswith("2.")
# 
# if PYDANTIC_V2:
#     from pydantic.v1 <import stuff>
# 
# else:
#     from pydantic <import stuff>
sed -i -E '/^from pydantic import /{s/^from pydantic import (.*)$/import pydantic\nPYDANTIC_V2 = hasattr(pydantic, "VERSION") and pydantic.VERSION.startswith("2.")\nif PYDANTIC_V2:\n    from pydantic.v1 import \1  # noqa: F401\nelse:\n    from pydantic import \1  # type: ignore # noqa: F401/}' ${CLIENT_DIR}/openai.py