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
RUN useradd -m llmengine -s /bin/bash
RUN chown -R llmengine /venv
RUN chown -R llmengine /app
# Limits for nproc and consequently number of files open
ADD llm_engine/llm_engine/inference/limits.conf /etc/security/limits.conf
USER llmengine

RUN mkdir -p /app/ml_infra_core/llm_engine.core
RUN chown -R llmengine /app/ml_infra_core

COPY --chown=llmengine ml_infra_core/llm_engine.core/requirements.txt ml_infra_core/llm_engine.core/requirements.txt
RUN --mount=type=secret,id=codeartifact-pip-conf,target=/etc/pip.conf,mode=0444 \
    PIP_CONFIG_FILE=/kaniko/pip/codeartifact_pip_conf \
    pip install -r ml_infra_core/llm_engine.core/requirements.txt --no-cache-dir
COPY --chown=llmengine ml_infra_core/llm_engine.core ml_infra_core/llm_engine.core
RUN pip install -e ml_infra_core/llm_engine.core

RUN mkdir -p /app/insight
RUN chown -R llmengine /app/insight

COPY --chown=llmengine insight/client/requirements.txt insight/client/requirements.txt
RUN pip install -r insight/client/requirements.txt --no-cache-dir
COPY --chown=llmengine insight/client insight/client
RUN pip install -e insight/client

# Not good for layer caching oh well
# The inference code should only need these few files/directories to function (hopefully)
# Don't copy the entire folder for security reasons

RUN mkdir -p /app/llm_engine
RUN mkdir -p /app/llm_engine/llm_engine

RUN chown -R llmengine /app/llm_engine

COPY --chown=llmengine \
    llm_engine/llm_engine/inference/requirements_base.txt \
    /app/llm_engine/llm_engine/inference/requirements_base.txt
RUN pip install -r /app/llm_engine/llm_engine/inference/requirements_base.txt

COPY --chown=llmengine llm_engine/setup.py /app/llm_engine/setup.py
COPY --chown=llmengine llm_engine/llm_engine.egg-info /app/llm_engine/llm_engine.egg-info
COPY --chown=llmengine llm_engine/llm_engine/__init__.py /app/llm_engine/llm_engine/__init__.py
COPY --chown=llmengine llm_engine/llm_engine/common /app/llm_engine/llm_engine/common
COPY --chown=llmengine llm_engine/llm_engine/domain /app/llm_engine/llm_engine/domain
COPY --chown=llmengine llm_engine/llm_engine/infra /app/llm_engine/llm_engine/infra
COPY --chown=llmengine llm_engine/llm_engine/inference /app/llm_engine/llm_engine/inference
WORKDIR /app/llm_engine
RUN pip install -e .
WORKDIR /app
