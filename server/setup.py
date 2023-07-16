# To get circleci to work
from setuptools import find_packages, setup

setup(
    name="scale-llm-engine-server",
    version="1.0.0",
    packages=[p for p in find_packages() if "tests" not in p],
    install_requires=[],
    entry_points={
        "console_scripts": [
            "start-service-builder=spellbook_serve.start_service_builder:entrypoint",
            "start-server=spellbook_serve.start_server:entrypoint",
            "start-fastapi-server=spellbook_serve.entrypoints.start_fastapi_server:entrypoint",
            "start-batch-job-orchestration=spellbook_serve.entrypoints.start_batch_job_orchestration:entrypoint",
            "hosted-inference-server=spellbook_serve.entrypoints.hosted_inference_server:entrypoint",
            "autogen=spellbook_serve.scripts.autogenerate_client_and_docs:entrypoint",
        ],
    },
)
