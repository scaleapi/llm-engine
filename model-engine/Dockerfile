# syntax = docker/dockerfile:experimental

# --- Builder: compile C extensions (pycurl, etc.) and install Python packages ---
FROM python:3.13-slim AS builder

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  gcc \
  build-essential \
  libssl-dev \
  libcurl4-openssl-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip install pip==24.2 setuptools
RUN pip install awscli==1.34.28 --no-cache-dir

WORKDIR /workspace/model-engine/
COPY model-engine/requirements-test.txt requirements-test.txt
COPY model-engine/requirements.txt requirements.txt
COPY model-engine/requirements_override.txt requirements_override.txt
RUN pip install -r requirements-test.txt --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir
# NOTE: aioboto3==10.4.0 -> aiobotocore==2.4.2 -> urllib3<1.27, which downgrades urllib3
# from 2.x back to 1.26.x. CVE-2023-43804, CVE-2023-45803, CVE-2024-37891 remain.
# Fix: upgrade aioboto3 to >=15.x (separate PR — breaking API changes).
RUN pip install -r requirements_override.txt --no-cache-dir
COPY model-engine/setup.py setup.py
COPY model-engine/model_engine_server model_engine_server
RUN pip install -e .

# --- Runtime: no build tools (eliminates linux-libc-dev and python3.13 CVEs) ---
FROM python:3.13-slim AS model-engine

WORKDIR /workspace

# Runtime-only system deps (vim omitted: multiple unpatched HIGH CVEs in Debian 13.4)
RUN apt-get update && apt-get install -y --no-install-recommends \
  dumb-init \
  git \
  openssh-client \
  curl \
  procps \
  htop \
  libcurl4 \
  && rm -rf /var/lib/apt/lists/*

# Install aws-iam-authenticator (architecture-aware)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
      curl -fLo /bin/aws-iam-authenticator https://github.com/kubernetes-sigs/aws-iam-authenticator/releases/download/v0.7.11/aws-iam-authenticator_0.7.11_linux_arm64; \
    else \
      curl -fLo /bin/aws-iam-authenticator https://github.com/kubernetes-sigs/aws-iam-authenticator/releases/download/v0.7.11/aws-iam-authenticator_0.7.11_linux_amd64; \
    fi && \
    chmod +x /bin/aws-iam-authenticator

# Install kubectl (architecture-aware)
RUN ARCH=$(uname -m | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/') && \
    curl -fLO "https://dl.k8s.io/release/v1.35.3/bin/linux/${ARCH}/kubectl" && \
    chmod +x kubectl && \
    mv kubectl /usr/local/bin/kubectl

# Copy Python packages, entry-point scripts, and source tree from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /workspace/model-engine /workspace/model-engine

RUN useradd --create-home --shell /bin/bash nonroot && \
    chown -R nonroot:nonroot /workspace

COPY integration_tests /workspace/integration_tests

WORKDIR /workspace
ENV PYTHONPATH /workspace
ENV WORKSPACE /workspace

USER nonroot
EXPOSE 5000
