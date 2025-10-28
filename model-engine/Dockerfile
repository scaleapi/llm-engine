FROM cgr.dev/scale.com/python-fips:3.10.19-dev
WORKDIR /workspace
USER root

RUN apk update && apk add \
            htop \
            dumb-init \
            libssh \
            openssh-client \
            iftop \
            curl \
            curl-dev \
            procps \
            libcurl-openssl4 \
            vim \
            kubectl \
            jq \
            gcc \
            glibc-dev \
            python-3.10-dev \
            libffi-dev \
            openssl-dev \
            build-base \
            postgresql-dev \
            libpq-16

RUN curl -Lo /bin/aws-iam-authenticator https://github.com/kubernetes-sigs/aws-iam-authenticator/releases/download/v0.5.9/aws-iam-authenticator_0.5.9_linux_amd64
RUN chmod +x /bin/aws-iam-authenticator

RUN pip install pip==24.2
RUN chmod -R 777 /workspace

RUN pip install awscli==1.34.28 --no-cache-dir

COPY model-engine/sitecustomize.py /usr/lib/python3.10/site-packages/sitecustomize.py

WORKDIR /workspace/model-engine/
COPY model-engine/requirements-test.txt /workspace/model-engine/requirements-test.txt
COPY model-engine/requirements.txt /workspace/model-engine/requirements.txt
COPY model-engine/requirements_override.txt /workspace/model-engine/requirements_override.txt
RUN pip install -r requirements-test.txt --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_override.txt --no-cache-dir
COPY model-engine/setup.py /workspace/model-engine/setup.py
COPY model-engine/model_engine_server /workspace/model-engine/model_engine_server
#COPY model-engine/service_configs /workspace/model-engine/service_configs
RUN pip install -e .

COPY integration_tests /workspace/integration_tests

WORKDIR /workspace
ENV PYTHONPATH=/workspace
ENV WORKSPACE=/workspace

EXPOSE 5000
