# Model Engine

The Model Engine is an API server that allows users to create, deploy, edit,
and delete machine learning endpoints. It consists of two main architectural
components:

- The [gateway](./model_engine_server/entrypoints/start_fastapi_server.py)
  provides a REST API for users to interact with. The routes of the REST API are
  defined in [`model_engine_server.api`](./model_engine_server/api).
- The [`model_engine_server.service_builder`](./model_engine_server/service_builder)
  package is the part of the code that creates the inference pods. It is the
  endpoint builder. When we do a `POST` request to `/endpoints`, this gets run.
  It gets run when users create or edit endpoints with `[POST,PUT] /v1/model-endpoints`

There are two other microservices:

- The [kubernetes cache](./model_engine_server/entrypoints/k8s_cache.py)
  stores endpoint metadata on Redis so that Model Engine does not overload the API
  server.
- The celery autoscaler (link TBD) automatically scales
  the number of inference pods based on the number of requests for async endpoints.

## Getting started

Be sure to install the global `../requirements-dev.txt` first prior
to any installations of requirements in this directory
(`pip install -r ../requirements-dev.txt`), as well as the pre-commit hooks
(`pre-commit install` in the `llm-engine` root folder). Then, install the
requirements files and this folder as editable

```bash
pip install -r requirements.txt && \
    pip install -r requirements-test.txt && \
    pip install -r requirements_override.txt && \
    pip install -e .
```

Run `mypy . --install-types` to set up mypy.

## Testing

Most of the business logic in Model Engine should contain unit tests, located in
[`tests/unit`](./tests/unit). To run the tests, run `pytest`.

Make sure to set a WORKSPACE env var that points to the root directory of this repo

```bash
export WORKSPACE=<path/to/llm-engine>
```
