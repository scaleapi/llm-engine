#!/bin/bash

set -eo pipefail

# Build and push Bert docker image to AWS ECR.
#
# Usage: ./build_and_upload_image.sh <AWS_ACCOUNT_ID> <IMAGE_TAG> bert

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_DIR=$SCRIPT_DIR/../../../..
DOCKERFILE=$PROJECT_DIR/model-engine/model_engine_server/inference/bert/Dockerfile.bert

if [ -z "$1" ]; then
  echo "Must supply AWS account ID"
  exit 1;
fi

if [ -z "$2" ]; then
  echo "Must supply the image tag"
  exit 1;
fi

if [ -z "$3" ]; then
  echo "Must supply the build target (either bert)"
  exit 1;
fi

ACCOUNT=$1
IMAGE_TAG=$2
BUILD_TARGET=$3
VLLM_VERSION=${VLLM_VERSION:-"0.5.3.post1"}

IMAGE="local/bert:$IMAGE_TAG"

# aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.us-west-2.amazonaws.com
DOCKER_BUILDKIT=1 docker build \
  -f ${DOCKERFILE} \
  --target ${BUILD_TARGET} \
  -t $IMAGE ${PROJECT_DIR}
# docker push $IMAGE
