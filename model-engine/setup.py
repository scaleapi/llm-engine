# To get circleci to work
from setuptools import find_packages, setup
setup(
    name="model_engine_server",
    version="1.0.0",
    packages=[p for p in find_packages() if "tests" not in p],
    install_requires=[],
    entry_points={
        "console_scripts": [
            "start-service-builder=model_engine_server.start_service_builder:entrypoint",
            "start-server=model_engine_server.start_server:entrypoint",
            "start-fastapi-server=model_engine_server.entrypoints.start_fastapi_server:entrypoint",
            "start-batch-job-orchestration=model_engine_server.entrypoints.start_batch_job_orchestration:entrypoint",
            "hosted-inference-server=model_engine_server.entrypoints.hosted_inference_server:entrypoint",
            "autogen=model_engine_server.scripts.autogenerate_client_and_docs:entrypoint",
            "launch-admin=model_engine_server.cli.bin:entrypoint",
        ],
    }
)
