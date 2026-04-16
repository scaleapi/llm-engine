# syntax = docker/dockerfile:1

FROM cgr.dev/chainguard/python:latest-dev AS builder

USER root
WORKDIR /workspace

RUN apk add --no-cache \
  bash \
  build-base \
  cmake \
  curl \
  curl-dev \
  dumb-init \
  git \
  openssl-dev

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

FROM cgr.dev/chainguard/python:latest AS model-engine

USER root
WORKDIR /workspace

COPY --from=builder --chown=nonroot:nonroot /workspace/venv /workspace/venv
COPY --from=builder --chown=nonroot:nonroot /workspace/model-engine /workspace/model-engine
COPY --from=builder /usr/bin/dumb-init /usr/bin/dumb-init
COPY --from=builder /usr/bin/git /usr/bin/git
COPY --from=builder /usr/libexec/git-core /usr/libexec/git-core
COPY --from=builder /usr/lib/libpcre2-8.so.0 /usr/lib/libpcre2-8.so.0
COPY --from=builder /usr/lib/libpcre2-8.so.0.15.0 /usr/lib/libpcre2-8.so.0.15.0
COPY --from=builder /usr/lib/libz.so.1 /usr/lib/libz.so.1
COPY --from=builder /usr/lib/libz.so.1.3.2 /usr/lib/libz.so.1.3.2

ENV PATH="/workspace/venv/bin:/usr/local/bin:/usr/libexec/git-core:/usr/bin"
ENV PYTHONPATH=/workspace
ENV WORKSPACE=/workspace

USER nonroot
EXPOSE 5000
