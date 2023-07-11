install:
	pip install -r requirements.txt
	pip install -r requirements_override.txt
	pip install -e .

install-test:
	pip install -r requirements-test.txt

install-dev:
	pip install -r requirements-dev.txt

requirements: install-dev
	pip-compile --allow-unsafe --no-emit-index-url --no-emit-trusted-host --output-file=requirements.txt requirements.in

install-all: install install-test install-dev

test:
	WORKSPACE=.. pytest
