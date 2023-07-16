ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG REQUIREMENTS_FILE
COPY --chown=llmengine ${REQUIREMENTS_FILE} /app/llm_engine/llm_engine/inference/requirements.txt
RUN PIP_CONFIG_FILE=/kaniko/pip/codeartifact_pip_conf pip install -r /app/llm_engine/llm_engine/inference/requirements.txt

ENV PYTHONPATH /app
