ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app

RUN rm -rf /var/lib/apt/lists/*

# Install basic packages.
RUN apt-get update && apt-get install -y \
      apt-utils \
      dumb-init \
      git \
      ssh \
      emacs-nox \
      htop \
      iftop \
      vim \
      libsm6 \
      libxext6 \
      libcurl4-openssl-dev \
      libssl-dev \
      python3-dev \
      gcc \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=root model-engine /app/model-engine
WORKDIR /app/model-engine
RUN pip install -e .
WORKDIR /app

RUN pip install -r /app/model-engine/model_engine_server/inference/requirements_base.txt
