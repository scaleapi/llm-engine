FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN pip uninstall torch -y
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt
RUN wget https://github.com/peak/s5cmd/releases/download/v2.2.1/s5cmd_2.2.1_Linux-64bit.tar.gz
RUN tar -xvzf s5cmd_2.2.1_Linux-64bit.tar.gz
COPY vllm_server.py /workspace/vllm_server.py
