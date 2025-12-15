# syntax = docker/dockerfile:experimental

FROM python:3.10.15-slim as model-engine
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
  apt-utils \
  dumb-init \
  git \
  ssh \
  htop \
  iftop \
  vim \
  curl \
  procps \
  libcurl4-openssl-dev \
  libssl-dev \
  python3-dev \
  gcc \
  build-essential \
  telnet \
  && rm -rf /var/lib/apt/lists/*

# Install aws-iam-authenticator (architecture-aware)
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ]; then \
      curl -Lo /bin/aws-iam-authenticator https://github.com/kubernetes-sigs/aws-iam-authenticator/releases/download/v0.5.9/aws-iam-authenticator_0.5.9_linux_arm64; \
    else \
      curl -Lo /bin/aws-iam-authenticator https://github.com/kubernetes-sigs/aws-iam-authenticator/releases/download/v0.5.9/aws-iam-authenticator_0.5.9_linux_amd64; \
    fi && \
    chmod +x /bin/aws-iam-authenticator

# Install kubectl (architecture-aware)
RUN ARCH=$(dpkg --print-architecture) && \
    curl -LO "https://dl.k8s.io/release/v1.23.13/bin/linux/${ARCH}/kubectl" && \
    chmod +x kubectl && \
    mv kubectl /usr/local/bin/kubectl

# Pin pip version
RUN pip install pip==24.2
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
