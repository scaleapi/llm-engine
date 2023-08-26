#!/bin/bash

# Build and push vLLM docker image to ECR.

set -eo pipefail

if [ -z "$1" ]; then
  echo "Must supply the image tag"
  exit 1;
fi

IMAGE_TAG=$1
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 692474966980.dkr.ecr.us-west-2.amazonaws.com
DOCKER_BUILDKIT=1 docker build -t 692474966980.dkr.ecr.us-west-2.amazonaws.com/vllm:$IMAGE_TAG .
docker push 692474966980.dkr.ecr.us-west-2.amazonaws.com/vllm:$IMAGE_TAG
