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
RUN useradd -m modelengine -s /bin/bash
RUN chown -R modelengine /venv
RUN chown -R modelengine /app
# Limits for nproc and consequently number of files open
ADD model-engine/model_engine_server/inference/limits.conf /etc/security/limits.conf
USER modelengine

# Not good for layer caching oh well
# The inference code should only need these few files/directories to function (hopefully)
# Don't copy the entire folder for security reasons

RUN mkdir -p /app/model-engine
RUN mkdir -p /app/model-engine/model_engine_server

RUN chown -R modelengine /app/model-engine

COPY --chown=modelengine \
    model-engine/model_engine_server/inference/requirements_base.txt \
    /app/model-engine/model_engine_server/inference/requirements_base.txt
RUN pip install -r /app/model-engine/model_engine_server/inference/requirements_base.txt

COPY --chown=modelengine model-engine/setup.py /app/model-engine/setup.py
COPY --chown=modelengine model-engine/model_engine_server/__init__.py /app/model-engine/model_engine_server/__init__.py
COPY --chown=modelengine model-engine/model_engine_server/common /app/model-engine/model_engine_server/common
COPY --chown=modelengine model-engine/model_engine_server/core /app/model-engine/model_engine_server/core
COPY --chown=modelengine model-engine/model_engine_server/domain /app/model-engine/model_engine_server/domain
COPY --chown=modelengine model-engine/model_engine_server/infra /app/model-engine/model_engine_server/infra
COPY --chown=modelengine model-engine/model_engine_server/inference /app/model-engine/model_engine_server/inference
WORKDIR /app/model-engine
RUN pip install -e .

WORKDIR /app
