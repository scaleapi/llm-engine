#!/bin/bash

# Build and push batch inference vLLM docker image to AWS ECR.

set -eo pipefail

if [ -z "$1" ]; then
  echo "Must supply AWS account ID"
  exit 1;
fi

if [ -z "$2" ]; then
  echo "Must supply the image tag"
  exit 1;
fi

if [ -z "$3" ]; then
  echo "Must supply the repo name"
  exit 1;
fi

REPO_NAME=$3
IMAGE_TAG=$2
ACCOUNT=$1
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.us-west-2.amazonaws.com
DOCKER_BUILDKIT=1 docker build -t $ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/$REPO_NAME:$IMAGE_TAG -f Dockerfile_vllm ../../../../
docker push $ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/$REPO_NAME:$IMAGE_TAG
