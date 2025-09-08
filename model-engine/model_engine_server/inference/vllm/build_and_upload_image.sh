#!/bin/bash

set -eo pipefail

# Build vLLM docker image locally.
#
# Usage: VLLM_VERSION=0.10.0 ./build_and_upload_image.sh <IMAGE_TAG> vllm|vllm_batch|vllm_batch_v2

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_DIR=$SCRIPT_DIR/../../../..
DOCKERFILE=$PROJECT_DIR/model-engine/model_engine_server/inference/vllm/Dockerfile.vllm

if [ -z "$1" ]; then
  echo "Must supply the image tag"
  exit 1;
fi

if [ -z "$2" ]; then
  echo "Must supply the build target (either vllm or vllm_batch_v2)"
  exit 1;
fi

IMAGE_TAG=$1
BUILD_TARGET=$2
VLLM_VERSION=${VLLM_VERSION:-"0.10.0"}
VLLM_BASE_REPO=${VLLM_BASE_REPO:-"vllm/vllm-openai"}

# if build target = vllm use vllm otherwise use vllm_batch
if [ "$BUILD_TARGET" == "vllm" ]; then
  IMAGE=vllm-onprem:$IMAGE_TAG
else
  IMAGE=vllm-batch-onprem:$IMAGE_TAG
fi

echo "Building Docker image: $IMAGE"
echo "vLLM Version: $VLLM_VERSION"
echo "Build Target: $BUILD_TARGET"

DOCKER_BUILDKIT=1 docker build \
  --build-arg VLLM_VERSION=${VLLM_VERSION} \
  --build-arg VLLM_BASE_REPO=${VLLM_BASE_REPO} \
  -f ${DOCKERFILE} \
  --target ${BUILD_TARGET} \
  -t $IMAGE ${PROJECT_DIR}

echo "Successfully built: $IMAGE"
