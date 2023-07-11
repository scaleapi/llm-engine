ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG REQUIREMENTS_FILE
COPY --chown=scalelaunch ${REQUIREMENTS_FILE} /app/spellbook_serve/spellbook_serve/inference/requirements.txt
RUN PIP_CONFIG_FILE=/kaniko/pip/codeartifact_pip_conf pip install -r /app/spellbook_serve/spellbook_serve/inference/requirements.txt

ENV PYTHONPATH /app
