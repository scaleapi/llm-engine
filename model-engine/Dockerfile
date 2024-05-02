# syntax = docker/dockerfile:experimental

FROM python:3.8.18-slim as model-engine
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
  && rm -rf /var/lib/apt/lists/*

RUN curl -Lo /bin/aws-iam-authenticator https://github.com/kubernetes-sigs/aws-iam-authenticator/releases/download/v0.6.20/aws-iam-authenticator_0.6.20_linux_amd64
RUN chmod +x /bin/aws-iam-authenticator

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/v1.23.13/bin/linux/amd64/kubectl" \
  && chmod +x kubectl \
  && mv kubectl /usr/local/bin/kubectl

# Pin pip version
RUN pip install pip==24.0
RUN chmod -R 777 /workspace

# Install AWS CLI
RUN pip install awscli==1.25.62 --no-cache-dir

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
