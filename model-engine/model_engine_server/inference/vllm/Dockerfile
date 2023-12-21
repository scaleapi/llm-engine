FROM nvcr.io/nvidia/pytorch:23.09-py3

RUN pip uninstall torch -y
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

# install special version of megablocks
RUN pip install git+https://github.com/stanford-futuredata/megablocks.git@5897cd6f254b7b3edf7a708a3a3314ecb54b6f78#egg=megablocks

RUN wget https://github.com/peak/s5cmd/releases/download/v2.2.1/s5cmd_2.2.1_Linux-64bit.tar.gz
RUN tar -xvzf s5cmd_2.2.1_Linux-64bit.tar.gz

COPY vllm_server.py /workspace/vllm_server.py
