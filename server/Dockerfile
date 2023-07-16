# syntax = docker/dockerfile:experimental

FROM python:3.8.8-slim as llm-engine
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
  postgresql \
  telnet \
  && rm -rf /var/lib/apt/lists/*

RUN curl -Lo /bin/aws-iam-authenticator https://github.com/kubernetes-sigs/aws-iam-authenticator/releases/download/v0.5.9/aws-iam-authenticator_0.5.9_linux_amd64
RUN chmod +x /bin/aws-iam-authenticator

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/v1.17.9/bin/linux/amd64/kubectl" \
  && chmod +x kubectl \
  && mv kubectl /usr/local/bin/kubectl

# Pin pip version
RUN pip install pip==23.0.1
RUN chmod -R 777 /workspace

## grab llm_engine project (w/ requirements install layer caching)
WORKDIR /workspace/llm_engine/
COPY llm_engine/requirements-test.txt /workspace/llm_engine/requirements-test.txt
COPY llm_engine/requirements.txt /workspace/llm_engine/requirements.txt
COPY llm_engine/requirements_override.txt /workspace/llm_engine/requirements_override.txt
RUN pip install -r requirements-test.txt --no-cache-dir
RUN --mount=type=secret,id=codeartifact-pip-conf,target=/etc/pip.conf \
    PIP_CONFIG_FILE=/kaniko/pip/codeartifact_pip_conf \
    pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_override.txt --no-cache-dir
COPY llm_engine/pyproject.toml /workspace/llm_engine/pyproject.toml
COPY llm_engine/setup.py /workspace/llm_engine/setup.py
COPY llm_engine/llm_engine /workspace/llm_engine/llm_engine
RUN pip install -e .

WORKDIR /workspace
ENV PYTHONPATH /workspace
ENV WORKSPACE /workspace

EXPOSE 5000
EXPOSE 5001
EXPOSE 5002
EXPOSE 5005

RUN useradd -m user -s /bin/bash
USER user
