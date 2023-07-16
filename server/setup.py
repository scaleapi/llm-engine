# To get circleci to work
from setuptools import find_packages, setup

setup(
    name="scale-llm-engine-server",
    version="1.0.0",
    packages=[p for p in find_packages() if "tests" not in p],
    install_requires=[],
    entry_points={
        "console_scripts": [
            "start-service-builder=llm_engine_server.start_service_builder:entrypoint",
            "start-server=llm_engine_server.start_server:entrypoint",
            "start-fastapi-server=llm_engine_server.entrypoints.start_fastapi_server:entrypoint",
            "start-batch-job-orchestration=llm_engine_server.entrypoints.start_batch_job_orchestration:entrypoint",
            "hosted-inference-server=llm_engine_server.entrypoints.hosted_inference_server:entrypoint",
            "autogen=llm_engine_server.scripts.autogenerate_client_and_docs:entrypoint",
        ],
    },
)
