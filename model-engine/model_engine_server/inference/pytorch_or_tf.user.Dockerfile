ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG REQUIREMENTS_FILE
COPY --chown=modelengine ${REQUIREMENTS_FILE} /app/model-engine/model_engine_server/inference/requirements.txt
RUN --mount=type=secret,id=codeartifact-pip-conf,target=/etc/pip.conf,mode=0444 \ 
    PIP_CONFIG_FILE=/kaniko/pip/codeartifact_pip_conf \
    pip install -r /app/model-engine/model_engine_server/inference/requirements.txt

ENV PYTHONPATH /app
