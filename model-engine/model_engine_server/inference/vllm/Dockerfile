FROM nvcr.io/nvidia/pytorch:23.09-py3

RUN apt-get update \
    && apt-get install -y \
    gdb \
    psmisc \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip uninstall torch -y

# Need to fix flash-infer at 0.0.8 to support gemma models
# See https://github.com/vllm-project/vllm/issues/7060#issuecomment-2266248014
RUN pip uninstall flash-infer -y
RUN pip install flash-infer==0.0.8 --index-url -i https://flashinfer.ai/whl/cu121/torch2.1/

COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

RUN wget https://github.com/peak/s5cmd/releases/download/v2.2.1/s5cmd_2.2.1_Linux-64bit.tar.gz
RUN tar -xvzf s5cmd_2.2.1_Linux-64bit.tar.gz

COPY vllm_server.py /workspace/vllm_server.py
