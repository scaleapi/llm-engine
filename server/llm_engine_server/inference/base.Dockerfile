ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app

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

COPY --chown=root llm_engine /app/llm_engine
WORKDIR /app/llm_engine
RUN pip install -e .
WORKDIR /app

RUN pip install -r /app/llm_engine/llm_engine/inference/requirements_base.txt
