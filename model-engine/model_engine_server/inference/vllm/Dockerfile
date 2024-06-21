#################### BASE BUILD IMAGE ####################
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS dev
RUN apt-get update -y \
    && apt-get install -y python3-pip git
# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-12.1/compat/
WORKDIR /workspace

COPY requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-build.txt
#################### BASE BUILD IMAGE ####################

#################### FLASH_ATTENTION Build IMAGE ####################
FROM dev as flash-attn-builder
# max jobs used for build
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
# flash attention version
ARG flash_attn_version=v2.4.2
ENV FLASH_ATTN_VERSION=${flash_attn_version}

WORKDIR /usr/src/flash-attention-v2

# Download the wheel or build it if a pre-compiled release doesn't exist
RUN pip --verbose wheel flash-attn==${FLASH_ATTN_VERSION} \
    --no-build-isolation --no-deps --no-cache-dir

#################### FLASH_ATTENTION Build IMAGE ####################

#################### Runtime IMAGE ####################
FROM nvcr.io/nvidia/pytorch:23.09-py3

RUN apt-get update \
    && apt-get install -y \
        gdb \
        psmisc \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Install flash attention (from pre-built wheel)
RUN --mount=type=bind,from=flash-attn-builder,src=/usr/src/flash-attention-v2,target=/usr/src/flash-attention-v2 \
    pip install /usr/src/flash-attention-v2/*.whl --no-cache-dir

RUN pip uninstall torch -y
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

RUN wget https://github.com/peak/s5cmd/releases/download/v2.2.1/s5cmd_2.2.1_Linux-64bit.tar.gz
RUN tar -xvzf s5cmd_2.2.1_Linux-64bit.tar.gz

COPY vllm_server.py /workspace/vllm_server.py

#################### Runtime IMAGE ####################
