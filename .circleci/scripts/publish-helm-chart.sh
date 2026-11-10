#!/usr/bin/env bash
# Publish the model-engine Helm chart to the public ECR registry.
#
# Idempotent: reads the chart version from charts/model-engine/Chart.yaml and
# only packages + pushes when that version tag is not already published. So it
# is safe to run on every merge to main — an actual publish happens only when
# the chart version was bumped.
#
# Requires: helm, aws-cli, and AWS credentials for a principal in account
# 692474966980 with ecr-public publish permission on
# model-engine-helm-charts/model-engine (the circleci-llm-engine-chart-publish
# OIDC role).
set -euo pipefail

CHART_DIR="charts/model-engine"
PUBLIC_ECR_REGISTRY="public.ecr.aws/b2z8n5q1"
CHART_REPO_PATH="model-engine-helm-charts"          # helm appends the chart name (model-engine)
ECR_REPOSITORY_NAME="model-engine-helm-charts/model-engine"
ECR_PUBLIC_REGION="us-east-1"                        # ECR Public API only exists in us-east-1

VERSION="$(helm show chart "${CHART_DIR}" | awk '/^version:/ {print $2}')"
if [[ -z "${VERSION}" ]]; then
  echo "ERROR: could not determine chart version from ${CHART_DIR}/Chart.yaml" >&2
  exit 1
fi
echo "model-engine chart version: ${VERSION}"

# Never overwrite a published version. Only ImageNotFoundException means the
# tag is free; any other describe error fails the job instead of pushing.
if DESCRIBE_OUTPUT="$(aws ecr-public describe-images \
      --region "${ECR_PUBLIC_REGION}" \
      --repository-name "${ECR_REPOSITORY_NAME}" \
      --image-ids imageTag="${VERSION}" 2>&1)"; then
  echo "Chart ${VERSION} is already published to ${PUBLIC_ECR_REGISTRY}/${ECR_REPOSITORY_NAME}; refusing to push."
  exit 0
fi
if [[ "${DESCRIBE_OUTPUT}" != *ImageNotFoundException* ]]; then
  echo "ERROR: could not check whether chart ${VERSION} is already published:" >&2
  echo "${DESCRIBE_OUTPUT}" >&2
  exit 1
fi

echo "Publishing model-engine chart ${VERSION}..."
aws ecr-public get-login-password --region "${ECR_PUBLIC_REGION}" \
  | helm registry login --username AWS --password-stdin public.ecr.aws

PKG_DIR="$(mktemp -d)"
helm package "${CHART_DIR}" -d "${PKG_DIR}"
helm push "${PKG_DIR}/model-engine-${VERSION}.tgz" "oci://${PUBLIC_ECR_REGISTRY}/${CHART_REPO_PATH}"

echo "Published oci://${PUBLIC_ECR_REGISTRY}/${ECR_REPOSITORY_NAME}:${VERSION}"
