### THIS FILE IS DEPRECATED IN V1. INSTEAD, USE pytorch_or_tf.base.Dockerfile
### and pytorch_or_tf.user.Dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app

# Install basic packages.
# TODO: ffmpeg, libsm6, and lixext6 are essentially hardcoded from lidar.
# It's probably more correct to add support for arbitrary user-specified base images,
# otherwise this base image gets bloated over time.
RUN apt-get update && apt-get install -y \
      apt-utils \
      dumb-init \
      git \
      ssh \
      emacs-nox \
      htop \
      iftop \
      vim \
      ffmpeg \
      libsm6 \
      libxext6 \
      libcurl4-openssl-dev \
      libssl-dev \
      python3-dev \
      gcc \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Apparently wget has a vulnerability so we remove it here
RUN apt-get remove wget -y

# Create a virtualenv for python so we install our packages in the right place
# Not sure how useful the existing contents of the pytorch image are anymore :/ Maybe it's used for cuda/cudnn installs
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Run everything as not-root user
RUN useradd -m scalelaunch -s /bin/bash
RUN chown -R scalelaunch /venv
RUN chown -R scalelaunch /app
# Limits for nproc and consequently number of files open
ADD spellbook_serve/spellbook_serve/inference/limits.conf /etc/security/limits.conf
USER scalelaunch

RUN mkdir -p /app/ml_infra_core/spellbook_serve.core
RUN chown -R scalelaunch /app/ml_infra_core

COPY --chown=scalelaunch ml_infra_core/spellbook_serve.core/requirements.txt ml_infra_core/spellbook_serve.core/requirements.txt
RUN PIP_CONFIG_FILE=/kaniko/pip/codeartifact_pip_conf pip install -r ml_infra_core/spellbook_serve.core/requirements.txt --no-cache-dir
COPY --chown=scalelaunch ml_infra_core/spellbook_serve.core ml_infra_core/spellbook_serve.core
RUN pip install -e ml_infra_core/spellbook_serve.core

RUN mkdir -p /app/insight
RUN chown -R scalelaunch /app/insight

COPY --chown=scalelaunch insight/client/requirements.txt insight/client/requirements.txt
RUN pip install -r insight/client/requirements.txt --no-cache-dir
COPY --chown=scalelaunch insight/client insight/client
RUN pip install -e insight/client

# Not good for layer caching oh well
# The inference code should only need these few files/directories to function (hopefully)
# Don't copy the entire folder for security reasons

RUN mkdir -p /app/spellbook_serve
RUN mkdir -p /app/spellbook_serve/spellbook_serve

RUN chown -R scalelaunch /app/spellbook_serve

COPY --chown=scalelaunch spellbook_serve/setup.py /app/spellbook_serve/setup.py
COPY --chown=scalelaunch spellbook_serve/spellbook_serve.egg-info /app/spellbook_serve/spellbook_serve.egg-info
COPY --chown=scalelaunch spellbook_serve/spellbook_serve/__init__.py /app/spellbook_serve/spellbook_serve/__init__.py
COPY --chown=scalelaunch spellbook_serve/spellbook_serve/common /app/spellbook_serve/spellbook_serve/common
COPY --chown=scalelaunch spellbook_serve/spellbook_serve/domain /app/spellbook_serve/spellbook_serve/domain
COPY --chown=scalelaunch spellbook_serve/spellbook_serve/infra /app/spellbook_serve/spellbook_serve/infra
COPY --chown=scalelaunch spellbook_serve/spellbook_serve/inference /app/spellbook_serve/spellbook_serve/inference
WORKDIR /app/spellbook_serve
RUN pip install -e .
WORKDIR /app

RUN pip install -r /app/spellbook_serve/spellbook_serve/inference/requirements_base.txt
ARG REQUIREMENTS_FILE
COPY --chown=scalelaunch ${REQUIREMENTS_FILE} /app/spellbook_serve/spellbook_serve/inference/requirements.txt
RUN PIP_CONFIG_FILE=/kaniko/pip/codeartifact_pip_conf pip install -r /app/spellbook_serve/spellbook_serve/inference/requirements.txt


ENV PYTHONPATH /app
