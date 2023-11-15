FROM nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3

COPY requirements.txt /workspace/requirements.txt
WORKDIR /workspace
RUN pip install -r requirements.txt

# Install s5cmd
RUN wget https://github.com/peak/s5cmd/releases/download/v2.2.1/s5cmd_2.2.1_Linux-64bit.tar.gz
RUN tar -xvzf s5cmd_2.2.1_Linux-64bit.tar.gz

COPY launch_triton_server.py /workspace/launch_triton_server.py
COPY triton_model_repo /workspace/model_repo