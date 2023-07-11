FROM python:3.8.5-slim

# Install basic debugging tools. It won't increase the image size too much.
RUN apt-get update && apt-get install -y \
    emacs-nox \
    htop \
    iftop \
    vim \
    dumb-init \
    curl \
    git \
    libcurl4-openssl-dev \
    libssl-dev \
    python3-dev \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY scaleml/requirements.txt /workspace/scaleml/requirements.txt
RUN PIP_CONFIG_FILE=/kaniko/pip/codeartifact_pip_conf pip install -r /workspace/scaleml/requirements.txt --no-cache-dir
COPY scaleml /workspace/scaleml
RUN pip install -e /workspace/scaleml

WORKDIR /workspace/ml_infra_core/spellbook_serve.core
COPY ml_infra_core/spellbook_serve.core/requirements.txt /workspace/ml_infra_core/spellbook_serve.core/requirements.txt
RUN PIP_CONFIG_FILE=/kaniko/pip/codeartifact_pip_conf pip install -r requirements.txt --no-cache-dir
COPY ml_infra_core/spellbook_serve.core /workspace/ml_infra_core/spellbook_serve.core
RUN pip install -e /workspace/ml_infra_core/spellbook_serve.core

WORKDIR /workspace/ml_infra_core/spellbook_serve.db
COPY ml_infra_core/spellbook_serve.db/_requirements.txt /workspace/ml_infra_core/spellbook_serve.db/_requirements.txt
COPY ml_infra_core/spellbook_serve.db/requirements.txt /workspace/ml_infra_core/spellbook_serve.db/requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
COPY ml_infra_core/spellbook_serve.db /workspace/ml_infra_core/spellbook_serve.db
RUN pip install -e /workspace/ml_infra_core/spellbook_serve.db

COPY insight/client/requirements.txt /workspace/insight/client/requirements.txt
RUN pip install -r /workspace/insight/client/requirements.txt --no-cache-dir
COPY insight/client /workspace/insight/client
RUN pip install -e /workspace/insight/client

COPY scripts_py3 /workspace/scripts_py3

WORKDIR /workspace/spellbook_serve
COPY spellbook_serve/requirements_override.txt /workspace/spellbook_serve/requirements_override.txt
COPY spellbook_serve/requirements.txt /workspace/spellbook_serve/requirements.txt
RUN PIP_CONFIG_FILE=/kaniko/pip/codeartifact_pip_conf pip install -r requirements.txt \
    && pip install -r requirements_override.txt
COPY spellbook_serve /workspace/spellbook_serve
RUN pip install -e .
# needed for maybe_refresh_codeartifact, pin to arbitrary recent version
RUN pip install awscli==1.25.62
# reset WORKDIR
WORKDIR /

RUN curl -LO "https://dl.k8s.io/release/v1.17.9/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/kubectl

ENV PYTHONPATH /workspace
ENV WORKSPACE /workspace
WORKDIR /workspace
