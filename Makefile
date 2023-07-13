install:
	pip install -r requirements.txt
	pip install -r requirements_override.txt
	pip install -e .

install-test:
	pip install -r requirements-test.txt

install-dev:
	pip install -r requirements-dev.txt

install-docs:
	pip install -r requirements-docs.txt
	pip install -e clients/python/

requirements: install-dev
	pip-compile --allow-unsafe --no-emit-index-url --no-emit-trusted-host --output-file=requirements.txt requirements.in

install-all: install install-test install-dev install-docs

test:
	WORKSPACE=.. pytest

autogen-templates:
	pushd charts && \
	helm template spellbook-serve spellbook-serve -f spellbook-serve/values_circleci.yaml \
	-s templates/service_template_config_map.yaml \
	--set message='# THIS FILE IS AUTOGENERATED USING `just autogen-templates`. PLEASE EDIT THE GOTEMPLATE FILE IN THE HELM CHART!!!' \
	> ../spellbook_serve/infra/gateways/resources/templates/service_template_config_map_circleci.yaml \
	&& popd

build:
	docker-compose build spellbook-serve

dev:
	# TODO: add env variables to make this work.
	docker-compose up spellbook-serve-gateway-dev spellbook-serve-service-builder-dev

build-docs:
	mkdocs build

dev-docs:
	mkdocs serve
