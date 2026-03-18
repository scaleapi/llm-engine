# syntax = docker/dockerfile:experimental

FROM python:3.12-slim as model-engine

WORKDIR /workspace

# Install system dependencies (vim omitted: multiple unpatched HIGH CVEs in Debian 13.4)
RUN apt-get update && apt-get install -y --no-install-recommends \
  dumb-init \
  git \
  openssh-client \
  curl \
  procps \
  htop \
  gcc \
  build-essential \
  libssl-dev \
  libcurl4-openssl-dev \
  python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Install aws-iam-authenticator (architecture-aware)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
      curl -Lo /bin/aws-iam-authenticator https://github.com/kubernetes-sigs/aws-iam-authenticator/releases/download/v0.7.11/aws-iam-authenticator_0.7.11_linux_arm64; \
    else \
      curl -Lo /bin/aws-iam-authenticator https://github.com/kubernetes-sigs/aws-iam-authenticator/releases/download/v0.7.11/aws-iam-authenticator_0.7.11_linux_amd64; \
    fi && \
    chmod +x /bin/aws-iam-authenticator

# Install kubectl (architecture-aware)
RUN ARCH=$(uname -m | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/') && \
    curl -LO "https://dl.k8s.io/release/v1.35.2/bin/linux/${ARCH}/kubectl" && \
    chmod +x kubectl && \
    mv kubectl /usr/local/bin/kubectl

# Pin pip version and install setuptools (needed for legacy packages like ddtrace on Python 3.12)
RUN pip install pip==24.2 setuptools
RUN chmod -R 777 /workspace

# Install AWS CLI
RUN pip install awscli==1.34.28 --no-cache-dir

## grab model_engine_server project (w/ requirements install layer caching)
WORKDIR /workspace/model-engine/
COPY model-engine/requirements-test.txt /workspace/model-engine/requirements-test.txt
COPY model-engine/requirements.txt /workspace/model-engine/requirements.txt
COPY model-engine/requirements_override.txt /workspace/model-engine/requirements_override.txt
RUN pip install -r requirements-test.txt --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_override.txt --no-cache-dir
COPY model-engine/setup.py /workspace/model-engine/setup.py
COPY model-engine/model_engine_server /workspace/model-engine/model_engine_server
RUN pip install -e .

COPY integration_tests /workspace/integration_tests

WORKDIR /workspace
ENV PYTHONPATH /workspace
ENV WORKSPACE /workspace

EXPOSE 5000
