"""
Contains functions for building and pushing Docker images. The main input is a ML service name, not a Dockerfile
directly. This script will, based on ML service conventions, find the correct location of the Dockerfile for a given
ML service.
"""

import base64
import logging
import os
import pathlib
import subprocess
import textwrap
from os.path import exists
from typing import Dict, Optional

import boto3
import click
import docker

from spellbook_serve.core.config import ml_infra_config
from spellbook_serve.core.loggers import make_logger

from .remote_build import MODELS_ROOT, build_remote_wrapper

logger = make_logger("spellbook_serve.core.docker.docker_image", log_level=logging.INFO)

REGISTRY_ID = "692474966980"
ECR_REGION = "us-west-2"
ECR_REPO = f"{REGISTRY_ID}.dkr.ecr.{ECR_REGION}.amazonaws.com"


def _get_aws_creds() -> Dict[str, str]:
    aws_profile = os.environ.get("AWS_PROFILE")
    creds = {}
    if aws_profile:
        sess = boto3.session.Session(profile_name=aws_profile)
        frozen_creds = sess.get_credentials().get_frozen_credentials()
        creds = {
            "AWS_ACCESS_KEY_ID": frozen_creds.access_key,
            "AWS_SECRET_ACCESS_KEY": frozen_creds.secret_key,
            "AWS_SESSION_TOKEN": frozen_creds.token,
        }

    if "AWS_ACCESS_KEY_ID" in os.environ:
        creds["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID")

    if "AWS_SECRET_ACCESS_KEY" in os.environ:
        creds["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if "AWS_SESSION_TOKEN" in os.environ:
        creds["AWS_SESSION_TOKEN"] = os.environ.get("AWS_SESSION_TOKEN")

    return creds


def _get_image_tag(image_tag: Optional[str] = None) -> str:
    if image_tag:
        logger.info(f"Using supplied image tag: {image_tag}")
        return image_tag
    git_image_tag = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    logger.info(f"Using current git commit hash as image tag: {git_image_tag}")
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]))
    if dirty:
        raise ValueError(
            textwrap.dedent(
                """\
                Dirty git repository state detected.
                If you want to build the image for the current repository state, then explicitly set --image-tag.
                NOTE: This is NOT RECOMMENDED. You should probably commit your changes and push to Github."""
            )
        )
    return git_image_tag


@click.group()
def cli():
    """
    High level docker commands
    """


@cli.command()
@click.option("--service-name", required=True, type=str, help="The name of the ML service")
@click.option(
    "--dockerfile",
    default="Dockerfile",
    show_default=True,
    type=str,
    help="An override for the Dockerfile path under the service directory",
)
@click.option(
    "--image-tag",
    type=str,
    help="The Docker image tag to use. Defaults to the (clean) git SHA",
)
@click.option(
    "--test-command",
    type=str,
    help="Command to be run after building the image, for testing purposes",
)
def build(
    service_name: str,
    dockerfile: str = "Dockerfile",
    image_tag: Optional[str] = None,
    test_command: Optional[str] = None,
) -> None:
    local_args = (
        locals()
    )  # Make sure not to do this after grabbing the AWS creds, so that we don't print them out.

    tag = _get_image_tag(image_tag)
    image = f"{ECR_REPO}/{service_name}:{tag}"

    local_args["image"] = image

    logger.info(f"build args: {local_args}")

    docker_client = docker.APIClient(base_url="unix://var/run/docker.sock")

    codeartifact_script_path = "../../scripts_py3/scale_scripts/exe/maybe_refresh_codeartifact.py"
    if not exists(os.path.join(MODELS_ROOT, codeartifact_script_path)):
        raise Exception(
            "maybe_refresh_codeartifact.py does not exist, please git clone the whole models repo"
        )

    subprocess.check_output(
        [
            "python",
            codeartifact_script_path,
            "--export",
            ".codeartifact-pip-conf",
        ],
        cwd=str(MODELS_ROOT),
    )

    subprocess.check_output(
        [
            "docker",
            "build",
            ".",
            "-f",
            os.path.join(service_name, dockerfile),
            "-t",
            image,
            "--secret",
            "id=codeartifact-pip-conf,src=.codeartifact-pip-conf",
        ],
        cwd=str(MODELS_ROOT),
        env={"DOCKER_BUILDKIT": "1", "BUILDKIT_PROGRESS": "plain"},
    )

    if test_command:
        logger.info(
            textwrap.dedent(
                f"""
            Testing with 'docker run' on the built image.
            ARGS: {test_command}
            (NOTE: Expecting the test command to terminate.  """
            )
        )
        home_dir = str(pathlib.Path.home())
        output = docker_client.containers.run(  # pylint:disable=no-member
            image=image,
            command=test_command,
            volumes={
                os.path.join(home_dir, ".aws"): {
                    "bind": "/root/.aws/config",
                    "mode": "ro",
                }
            },
            environment={
                "AWS_PROFILE": ml_infra_config().profile_ml_worker,
                "AWS_CONFIG_FILE": "/root/.aws/config",
            },
            remove=True,
        )
        logger.info(output.decode("utf-8"))
    else:
        logger.info("Not testing image for starting behavior")


@cli.command()
@click.option("--service-name", required=True, type=str, help="The name of the ML service")
@click.option(
    "--image-tag",
    type=str,
    help="The Docker image tag to use. Defaults to the (clean) git SHA",
)
def push(service_name: str, image_tag: Optional[str] = None) -> None:
    local_args = locals()
    logger.info(f"push args: {local_args}")
    docker_client = docker.from_env()

    ecr_client = boto3.client("ecr", region_name=ECR_REGION)
    token = ecr_client.get_authorization_token(registryIds=[REGISTRY_ID])
    username, password = (
        base64.b64decode(token["authorizationData"][0]["authorizationToken"]).decode().split(":")
    )

    output = docker_client.images.push(
        repository=f"{ECR_REPO}/{service_name}",
        tag=_get_image_tag(image_tag),
        auth_config={"username": username, "password": password},
        stream=True,
        decode=True,
    )
    for line in output:
        logger.info(line)


cli.add_command(build_remote_wrapper, name="remote")
