#!/bin/bash

set -eo pipefail

# Build and push vLLM docker image to AWS ECR.
#
# Usage: VLLM_VERSION=0.6.3 ./build_and_upload_image.sh <AWS_ACCOUNT_ID> <IMAGE_TAG> vllm|vllm_batch|vllm_batch_v2

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_DIR=$SCRIPT_DIR/../../../..
DOCKERFILE=$PROJECT_DIR/model_engine_server/inference/vllm/Dockerfile.vllm

if [ -z "$1" ]; then
  echo "Must supply AWS account ID"
  exit 1;
fi

if [ -z "$2" ]; then
  echo "Must supply the image tag"
  exit 1;
fi

if [ -z "$3" ]; then
  echo "Must supply the build target (either vllm or vllm_batch_v2)"
  exit 1;
fi


ACCOUNT=$1
IMAGE_TAG=$2
BUILD_TARGET=$3
VLLM_VERSION=${VLLM_VERSION:-"0.6.2"}
VLLM_BASE_REPO=${VLLM_BASE_REPO:-"vllm/vllm-openai"}

# if build target = vllm use vllm otherwise use vllm_batch
if [ "$BUILD_TARGET" == "vllm" ]; then
  IMAGE=$ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/vllm:$IMAGE_TAG
else
  IMAGE=$ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/llm-engine/batch-infer-vllm:$IMAGE_TAG
fi

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.us-west-2.amazonaws.com
DOCKER_BUILDKIT=1 docker build \
  --build-arg VLLM_VERSION=${VLLM_VERSION} \
  --build-arg VLLM_BASE_REPO=${VLLM_BASE_REPO} \
  -f Dockerfile.vllm \
  --target ${BUILD_TARGET} \
  -t $IMAGE ${PROJECT_DIR}
docker push $IMAGE
