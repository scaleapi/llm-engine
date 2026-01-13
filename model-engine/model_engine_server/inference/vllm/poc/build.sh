#!/bin/bash
# Build and push the POC vLLM startup metrics image
#
# Usage:
#   ./build.sh                    # Build and push with default tag
#   ./build.sh mytag              # Build and push with custom tag

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
ECR_REGISTRY="692474966980.dkr.ecr.us-west-2.amazonaws.com"
IMAGE_NAME="vllm"
TAG="${1:-startup-metrics-poc-v1}"
FULL_IMAGE="${ECR_REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "Building POC image..."
echo "  Image: ${FULL_IMAGE}"

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Build the image
echo "Building Docker image..."
docker build -t "${FULL_IMAGE}" -f Dockerfile.poc .

# Push to ECR
echo "Pushing to ECR..."
docker push "${FULL_IMAGE}"

echo ""
echo "Build complete!"
echo "  Image: ${FULL_IMAGE}"
echo ""
echo "To deploy:"
echo "  kubectl apply -f deployment.yaml"
echo ""
echo "To check logs:"
echo "  kubectl logs -n scale-deploy -l app=vllm-startup-metrics-poc -f"
