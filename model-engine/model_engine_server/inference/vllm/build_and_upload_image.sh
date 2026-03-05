#!/bin/bash

set -eo pipefail

# Build and push vLLM docker image to AWS ECR.
#
# Usage: ./build_and_upload_image.sh <USER_TAG> <BUILD_TARGET> [OPTIONS]
#
# Required arguments:
#   USER_TAG          - User-provided tag suffix
#   BUILD_TARGET      - Build target: vllm, vllm_omni, vllm_batch, or vllm_batch_v2
#
# Optional flags:
#   --vllm-version=VERSION      - vLLM version (defaults to 0.16.0)
#   --vllm-omni-version=VERSION - vLLM Omni version (defaults to 0.16.0)
#   --aws-account-id=ID         - AWS account ID (defaults to 307185671274)
#   --vllm-base-repo=REPO       - Base Docker repository (defaults to vllm/vllm-openai)
#   --vllm-base-version=VERSION - Base image version (defaults to VLLM_VERSION)
#   --full-build                - Build vLLM base image from source using vLLM's own Dockerfile
#   --vllm-source-dir=PATH      - Path to local vLLM repo (required with --full-build)
#   --vllm-source-ref=REF       - Branch/tag/commit to checkout before building (e.g. releases/v0.16.0)
#   --cuda-arch=ARCHS           - CUDA architectures to compile for (e.g. '8.0 9.0'). Fewer = faster build.
#   --sccache-bucket=BUCKET     - S3 bucket for sccache (enables remote compilation cache)
#   --sccache-region=REGION     - AWS region for sccache bucket (defaults to us-west-2)
#   --vllm-omni-source-dir=PATH - Path to local vllm-omni repo (installs from source instead of PyPI)
#   --vllm-omni-source-ref=REF  - Branch/tag/commit to checkout in vllm-omni repo before building
#
# Environment variables (optional, can override flags):
#   VLLM_VERSION, VLLM_OMNI_VERSION, AWS_ACCOUNT_ID, VLLM_BASE_REPO
#   BUILDER - buildx builder name (defaults to hk_builder)
#   SCCACHE_BUCKET, SCCACHE_REGION - S3 sccache config
#
# The image tag will be automatically constructed as:
#   - For vllm_omni: {VLLM_VERSION}-omni-{VLLM_OMNI_VERSION}-{USER_TAG}
#   - For others: {VLLM_VERSION}-{USER_TAG}
#
# Examples:
#
#   # 1. Published versions (base image and pip version match)
#   ./build_and_upload_image.sh my-tag vllm
#   ./build_and_upload_image.sh my-tag vllm --vllm-version=0.15.1
#
#   # 2. Newer pip version on older base image (e.g. 0.16.0 wheel on 0.15.1 base)
#   ./build_and_upload_image.sh my-tag vllm --vllm-version=0.16.0 --vllm-base-version=0.15.1
#
#   # 3. Full build from vLLM source (when no published image or wheel exists)
#   ./build_and_upload_image.sh my-tag vllm --full-build --vllm-source-dir=/path/to/vllm --vllm-version=0.17.0
#
#   # 4. Full build from a specific branch/tag
#   ./build_and_upload_image.sh my-tag vllm --full-build --vllm-source-dir=/path/to/vllm --vllm-source-ref=releases/v0.16.0 --vllm-version=0.16.0
#
#   # 5. Full build targeting only H100 (much faster compilation)
#   ./build_and_upload_image.sh my-tag vllm --full-build --vllm-source-dir=/path/to/vllm --cuda-arch='9.0' --vllm-version=0.17.0
#
#   # 6. Full build with sccache for persistent compilation cache
#   ./build_and_upload_image.sh my-tag vllm --full-build --vllm-source-dir=/path/to/vllm --sccache-bucket=my-bucket --vllm-version=0.17.0
#
#   # 7. vllm_omni from published versions
#   ./build_and_upload_image.sh my-tag vllm_omni --vllm-version=0.16.0 --vllm-omni-version=0.16.0
#
#   # 8. vllm_omni with local vllm-omni source
#   ./build_and_upload_image.sh my-tag vllm_omni --vllm-version=0.16.0 --vllm-omni-source-dir=/path/to/vllm-omni
#
#   # 9. vllm_omni with newer vllm on older base + local omni source
#   ./build_and_upload_image.sh my-tag vllm_omni --vllm-version=0.16.0 --vllm-base-version=0.15.1 --vllm-omni-source-dir=/path/to/vllm-omni
#
#   # 10. Batch inference target
#   ./build_and_upload_image.sh my-tag vllm_batch --vllm-version=0.16.0 --vllm-base-version=0.15.1
#
#   # 11. Custom AWS account and base repo
#   ./build_and_upload_image.sh my-tag vllm --vllm-version=0.16.0 --account=123456789 --vllm-base-repo=my-registry/vllm-openai
#
#   # 12. Override parallelism for full builds
#   MAX_JOBS=64 NVCC_THREADS=32 ./build_and_upload_image.sh my-tag vllm --full-build --vllm-source-dir=/path/to/vllm --vllm-version=0.17.0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_DIR=$SCRIPT_DIR/../../../..
DOCKERFILE=$PROJECT_DIR/model-engine/model_engine_server/inference/vllm/Dockerfile.vllm

# Use buildx with persistent builder for cache mount persistence across builds
BUILDER=${BUILDER:-"hk_builder"}

# Default values
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-"307185671274"}
VLLM_VERSION=${VLLM_VERSION:-"0.16.0"}
VLLM_BASE_REPO=${VLLM_BASE_REPO:-"vllm/vllm-openai"}
VLLM_BASE_VERSION=""  # Will default to VLLM_VERSION if not set
FULL_BUILD=false
VLLM_SOURCE_DIR=""
VLLM_SOURCE_REF=""
CUDA_ARCH=""
SCCACHE_BUCKET=${SCCACHE_BUCKET:-""}
SCCACHE_REGION=${SCCACHE_REGION:-"us-west-2"}
VLLM_WHEEL_INDEX=""

VLLM_OMNI_VERSION=${VLLM_OMNI_VERSION:-"0.16.0"}
VLLM_OMNI_SOURCE_DIR=""
VLLM_OMNI_SOURCE_REF=""

if [ -z "$1" ]; then
  echo "Must supply the user-provided tag"
  exit 1;
fi

if [ -z "$2" ]; then
  echo "Must supply the build target (either vllm, vllm_omni or vllm_batch)"
  exit 1;
fi

USER_TAG=$1
BUILD_TARGET=$2

# Map flag names to variable names
declare -A FLAG_VARS=(
  ["--vllm-version"]="VLLM_VERSION"
  ["--vllm-omni-version"]="VLLM_OMNI_VERSION"
  ["--account"]="AWS_ACCOUNT_ID"
  ["--vllm-base-repo"]="VLLM_BASE_REPO"
  ["--vllm-base-version"]="VLLM_BASE_VERSION"
  ["--vllm-source-dir"]="VLLM_SOURCE_DIR"
  ["--vllm-source-ref"]="VLLM_SOURCE_REF"
  ["--cuda-arch"]="CUDA_ARCH"
  ["--sccache-bucket"]="SCCACHE_BUCKET"
  ["--sccache-region"]="SCCACHE_REGION"
  ["--vllm-omni-source-dir"]="VLLM_OMNI_SOURCE_DIR"
  ["--vllm-omni-source-ref"]="VLLM_OMNI_SOURCE_REF"
  ["--vllm-wheel-index"]="VLLM_WHEEL_INDEX"
)

# Parse keyword arguments
shift 2  # Remove USER_TAG and BUILD_TARGET
while [[ $# -gt 0 ]]; do
  flag="$1"
  
  # Handle boolean flags
  if [[ "$flag" == --full-build ]]; then
    FULL_BUILD=true
    shift
    continue
  fi
  
  # Extract flag name (handle both --flag and --flag=value)
  if [[ "$flag" == --*=* ]]; then
    flag_name="${flag%%=*}"
    value="${flag#*=}"
  else
    flag_name="$flag"
    value="$2"
    shift  # Will shift again after setting value
  fi
  
  var_name="${FLAG_VARS[$flag_name]}"
  if [ -z "$var_name" ]; then
    echo "Unknown option: $flag_name"
    echo "Use --help for usage information"
    exit 1
  fi
  
  if [[ "$flag" != --*=* ]] && [ -z "$value" ]; then
    echo "Error: $flag_name requires a value"
    exit 1
  fi
  
  eval "$var_name=\"$value\""
  shift
done

# Set base version to vllm version if not explicitly provided
if [ -z "$VLLM_BASE_VERSION" ]; then
  VLLM_BASE_VERSION="$VLLM_VERSION"
fi

# Known pre-release wheel indexes (version -> commit-specific wheel URL)
# Add entries here when a version has wheels but no PyPI release yet
declare -A VLLM_WHEEL_INDEXES=(
  ["0.16.0"]="https://wheels.vllm.ai/2d5be1dd5ce2e44dfea53ea03ff61143da5137eb"
)

# Required torch version for each vllm version (when base image has a different torch)
declare -A VLLM_TORCH_VERSIONS=(
  ["0.16.0"]="2.10.0"
)

# Auto-resolve wheel index if not explicitly provided
if [ -z "$VLLM_WHEEL_INDEX" ] && [ -n "${VLLM_WHEEL_INDEXES[$VLLM_VERSION]}" ]; then
  VLLM_WHEEL_INDEX="${VLLM_WHEEL_INDEXES[$VLLM_VERSION]}"
  echo "==> Auto-resolved wheel index for v${VLLM_VERSION}: $VLLM_WHEEL_INDEX"
fi

# Auto-resolve torch version if not explicitly provided
TORCH_VERSION=${TORCH_VERSION:-""}
if [ -z "$TORCH_VERSION" ] && [ -n "${VLLM_TORCH_VERSIONS[$VLLM_VERSION]}" ]; then
  TORCH_VERSION="${VLLM_TORCH_VERSIONS[$VLLM_VERSION]}"
  echo "==> Auto-resolved torch version for vllm v${VLLM_VERSION}: ${TORCH_VERSION}"
fi

# Validate --full-build requires --vllm-source-dir
if [ "$FULL_BUILD" = "true" ] && [ -z "$VLLM_SOURCE_DIR" ]; then
  echo "Error: --full-build requires --vllm-source-dir=<path>"
  exit 1
fi

if [ -n "$VLLM_SOURCE_DIR" ] && [ ! -d "$VLLM_SOURCE_DIR" ]; then
  echo "Error: --vllm-source-dir path does not exist: $VLLM_SOURCE_DIR"
  exit 1
fi

if [ -n "$VLLM_SOURCE_DIR" ] && [ ! -f "$VLLM_SOURCE_DIR/docker/Dockerfile" ]; then
  echo "Error: $VLLM_SOURCE_DIR does not look like a vLLM repo (missing docker/Dockerfile)"
  exit 1
fi

# Checkout the requested ref in the source dir
if [ -n "$VLLM_SOURCE_REF" ]; then
  if [ -z "$VLLM_SOURCE_DIR" ]; then
    echo "Error: --vllm-source-ref requires --vllm-source-dir"
    exit 1
  fi
  echo "==> Checking out $VLLM_SOURCE_REF in $VLLM_SOURCE_DIR"
  git -C "$VLLM_SOURCE_DIR" checkout "$VLLM_SOURCE_REF"
fi

if [ -n "$VLLM_OMNI_SOURCE_REF" ]; then
  if [ -z "$VLLM_OMNI_SOURCE_DIR" ]; then
    echo "Error: --vllm-omni-source-ref requires --vllm-omni-source-dir"
    exit 1
  fi
  echo "==> Checking out $VLLM_OMNI_SOURCE_REF in $VLLM_OMNI_SOURCE_DIR"
  git -C "$VLLM_OMNI_SOURCE_DIR" checkout "$VLLM_OMNI_SOURCE_REF"
fi

echo "VLLM_VERSION: $VLLM_VERSION"
echo "VLLM_OMNI_VERSION: $VLLM_OMNI_VERSION"
echo "VLLM_BASE_VERSION: $VLLM_BASE_VERSION"
echo "AWS_ACCOUNT_ID: $AWS_ACCOUNT_ID"
echo "VLLM_BASE_REPO: $VLLM_BASE_REPO"
echo "BUILD_TARGET: $BUILD_TARGET"
echo "USER_TAG: $USER_TAG"
echo "FULL_BUILD: $FULL_BUILD"
if [ -n "$VLLM_SOURCE_DIR" ]; then
  echo "VLLM_SOURCE_DIR: $VLLM_SOURCE_DIR"
fi
if [ -n "$VLLM_SOURCE_REF" ]; then
  echo "VLLM_SOURCE_REF: $VLLM_SOURCE_REF"
fi
if [ -n "$CUDA_ARCH" ]; then
  echo "CUDA_ARCH: $CUDA_ARCH"
fi
if [ -n "$VLLM_OMNI_SOURCE_DIR" ]; then
  echo "VLLM_OMNI_SOURCE_DIR: $VLLM_OMNI_SOURCE_DIR"
fi
if [ -n "$VLLM_OMNI_SOURCE_REF" ]; then
  echo "VLLM_OMNI_SOURCE_REF: $VLLM_OMNI_SOURCE_REF"
fi

# Construct image tag based on vllm version and user tag
if [ "$BUILD_TARGET" == "vllm_omni" ]; then
  IMAGE_TAG="${VLLM_VERSION}-omni-${VLLM_OMNI_VERSION}-${USER_TAG}"
else
  IMAGE_TAG="${VLLM_VERSION}-${USER_TAG}"
fi

# if build target = vllm use vllm otherwise use vllm_batch
if [ "$BUILD_TARGET" == "vllm" ] || [ "$BUILD_TARGET" == "vllm_omni" ]; then
  IMAGE=$AWS_ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/vllm:$IMAGE_TAG
else
  IMAGE=$AWS_ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/llm-engine/batch-infer-vllm:$IMAGE_TAG
fi

echo "Building and pushing image $IMAGE"

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com

# Auto-detect parallelism from machine resources
AVAILABLE_CORES=$(nproc)
# Effective parallelism is roughly MAX_JOBS * NVCC_THREADS, so divide conservatively
MAX_JOBS=${MAX_JOBS:-$(( AVAILABLE_CORES / 4 ))}
NVCC_THREADS=${NVCC_THREADS:-$(( AVAILABLE_CORES / 7 ))}
# Floor to at least 2
[ "$MAX_JOBS" -lt 2 ] && MAX_JOBS=2
[ "$NVCC_THREADS" -lt 2 ] && NVCC_THREADS=2

echo "==> Build parallelism: MAX_JOBS=$MAX_JOBS, NVCC_THREADS=$NVCC_THREADS (from $AVAILABLE_CORES cores)"

# Full build: build vLLM base image from source using their own Dockerfile
if [ "$FULL_BUILD" = "true" ]; then
  VLLM_LOCAL_IMAGE="vllm-local:${VLLM_VERSION}"
  echo "==> Building vLLM base image from source: $VLLM_SOURCE_DIR"
  echo "==> Local base image tag: $VLLM_LOCAL_IMAGE"

  FULL_BUILD_ARGS=(
    --progress=plain
    --build-arg max_jobs=${MAX_JOBS}
    --build-arg nvcc_threads=${NVCC_THREADS}
  )

  if [ -n "$CUDA_ARCH" ]; then
    FULL_BUILD_ARGS+=(--build-arg torch_cuda_arch_list="${CUDA_ARCH}")
  fi

  # sccache: remote S3 compilation cache (survives across machines)
  # Requires an S3 bucket and IAM credentials accessible from the build environment.
  # Without this, ccache (local to the buildx builder) is still used automatically.
  if [ -n "$SCCACHE_BUCKET" ]; then
    echo "==> sccache enabled: bucket=$SCCACHE_BUCKET region=$SCCACHE_REGION"
    FULL_BUILD_ARGS+=(
      --build-arg USE_SCCACHE=1
      --build-arg SCCACHE_BUCKET_NAME="${SCCACHE_BUCKET}"
      --build-arg SCCACHE_REGION_NAME="${SCCACHE_REGION}"
    )
  fi

  docker buildx build \
    --builder ${BUILDER} \
    --load \
    "${FULL_BUILD_ARGS[@]}" \
    -f "${VLLM_SOURCE_DIR}/docker/Dockerfile" \
    --target vllm-openai \
    -t "${VLLM_LOCAL_IMAGE}" \
    "${VLLM_SOURCE_DIR}"
fi

# Build docker build args
BUILD_ARGS=(
  --build-arg VLLM_VERSION=${VLLM_VERSION}
  --build-arg VLLM_BASE_REPO=${VLLM_BASE_REPO}
  --build-arg VLLM_BASE_VERSION=${VLLM_BASE_VERSION}
  --build-arg VLLM_OMNI_VERSION=${VLLM_OMNI_VERSION}
)

# Override the base image if we built one locally
if [ "$FULL_BUILD" = "true" ]; then
  BUILD_ARGS+=(--build-arg VLLM_BASE_IMAGE=${VLLM_LOCAL_IMAGE})
fi

# Pre-release wheel index
if [ -n "$VLLM_WHEEL_INDEX" ]; then
  BUILD_ARGS+=(--build-arg VLLM_WHEEL_INDEX="${VLLM_WHEEL_INDEX}")
fi

# Explicit torch version (needed when base image has incompatible torch)
if [ -n "$TORCH_VERSION" ]; then
  BUILD_ARGS+=(--build-arg TORCH_VERSION="${TORCH_VERSION}")
fi

# vllm-omni from local source
if [ -n "$VLLM_OMNI_SOURCE_DIR" ]; then
  if [ ! -d "$VLLM_OMNI_SOURCE_DIR" ]; then
    echo "Error: --vllm-omni-source-dir path does not exist: $VLLM_OMNI_SOURCE_DIR"
    exit 1
  fi
  BUILD_ARGS+=(
    --build-context vllm-omni-source="${VLLM_OMNI_SOURCE_DIR}"
    --build-arg VLLM_OMNI_FROM_SOURCE=true
  )
fi

echo "==> Building final image"

docker build \
  "${BUILD_ARGS[@]}" \
  -f ${DOCKERFILE} \
  --target ${BUILD_TARGET} \
  -t $IMAGE ${PROJECT_DIR}

docker push $IMAGE
