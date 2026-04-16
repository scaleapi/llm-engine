# syntax = docker/dockerfile:1

FROM cgr.dev/chainguard/python:latest-dev AS builder

USER root
WORKDIR /workspace
ARG TARGETARCH

RUN apk add --no-cache \
  bash \
  build-base \
  curl \
  curl-dev \
  dumb-init \
  git \
  go \
  openssl-dev \
  rsync

RUN python -m venv /workspace/venv
ENV PATH="/workspace/venv/bin:/usr/sbin:/usr/bin:/sbin:/bin"

WORKDIR /workspace/model-engine
COPY model-engine/requirements.txt requirements.txt
COPY model-engine/requirements_override.txt requirements_override.txt
RUN pip install --upgrade pip==24.2 setuptools cmake setuptools-rust
RUN pip install -r requirements.txt --no-cache-dir --no-build-isolation
RUN pip install -r requirements_override.txt --no-cache-dir
COPY model-engine/setup.py setup.py
COPY model-engine/model_engine_server model_engine_server
COPY model-engine/service_configs service_configs
RUN pip install -e .

RUN mkdir -p /tmp/runtime-bin /tmp/runtime-libs && \
  cp /usr/bin/dumb-init /tmp/runtime-bin/dumb-init && \
  cp /usr/bin/git /tmp/runtime-bin/git && \
  cp -R /usr/libexec/git-core /tmp/runtime-bin/git-core && \
  cp /usr/lib/libpcre2-8.so.0* /tmp/runtime-libs/ && \
  cp /usr/lib/libcurl.so.4* /tmp/runtime-libs/ && \
  cp /usr/lib/libz.so.1* /tmp/runtime-libs/ && \
  git clone --depth 1 --branch v1.35.3 https://github.com/kubernetes/kubernetes.git /tmp/k8s && \
  cd /tmp/k8s && \
  GOTOOLCHAIN=local make WHAT=cmd/kubectl && \
  cp _output/bin/kubectl /tmp/runtime-bin/kubectl && \
  GOBIN=/tmp/runtime-bin go install sigs.k8s.io/aws-iam-authenticator/cmd/aws-iam-authenticator@v0.7.11

FROM cgr.dev/chainguard/python:latest AS model-engine

USER root
WORKDIR /workspace

COPY --from=builder --chown=nonroot:nonroot /workspace/venv /workspace/venv
COPY --from=builder --chown=nonroot:nonroot /workspace/model-engine /workspace/model-engine
COPY --from=builder /tmp/runtime-bin/dumb-init /usr/bin/dumb-init
COPY --from=builder /tmp/runtime-bin/git /usr/bin/git
COPY --from=builder /tmp/runtime-bin/git-core /usr/libexec/git-core
COPY --from=builder /tmp/runtime-bin/kubectl /usr/local/bin/kubectl
COPY --from=builder /tmp/runtime-bin/aws-iam-authenticator /usr/local/bin/aws-iam-authenticator
COPY --from=builder /tmp/runtime-libs/ /usr/lib/

ENV PATH="/workspace/venv/bin:/usr/local/bin:/usr/libexec/git-core:/usr/bin"
ENV PYTHONPATH=/workspace
ENV WORKSPACE=/workspace

USER nonroot
EXPOSE 5000
