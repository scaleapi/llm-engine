import json
import os
import shutil
import subprocess
import tempfile
import uuid
from base64 import b64encode
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from string import Template
from subprocess import PIPE
from typing import Dict, Iterable, List, Optional, Union

import click
import tenacity
import yaml
from botocore.exceptions import ClientError, ProfileNotFound
from kubernetes import client
from kubernetes import config as kube_config
from kubernetes import watch
from kubernetes.config.config_exception import ConfigException
from model_engine_server.core.aws import storage_client
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger

logger = make_logger(logger_name())

S3_BUCKET = os.environ.get("S3_BUCKET", infra_config().s3_bucket)
SUB_BUCKET = "tmp/docker_contexts"
# Adjust if either this file or kaniko_template.yaml moves!
OWN_FILE_PATH = Path(__file__).resolve()
TEMPLATE_FILE = os.path.join(
    OWN_FILE_PATH.parent, os.getenv("KANIKO_TEMPLATE", "kaniko_template.yaml")
)
MODELS_ROOT = OWN_FILE_PATH.parents[4]

NAMESPACE = "default"

TIMEOUT_SECS = 1800


@dataclass
class BuildResult:
    """
    Status of a remote build process.
    """

    status: bool
    logs: str


def zip_context(
    s3_file_name: str,
    context: str,
    folders_to_include: List[str],
    ignore_file: Optional[str] = None,
) -> None:
    """
    Takes a path to a folder, zips up the folder and sticks it into s3

    :param s3_file_name: Bucket/file for context tar.gz, will upload to here
    :param context: Path to context for dockerfile, relative to calling script
    :param folders_to_include: List of paths to subfolders needed to build docker image, relative to context
    :param ignore_file: File (e.g. .dockerignore) containing things to ignore when preparing docker context.
        Relative to context. Contents of file are parsed according to tar's --exclude-from, which differs slightly from
        docker's behavior
    :return:
    """

    assert len(folders_to_include) > 0
    assert s3_file_name.endswith(".gz")
    s3_uri = f"s3://{S3_BUCKET}/{s3_file_name}"
    print(f"Uploading to s3 at: {s3_uri}")
    try:
        # Need to gimme_okta_aws_creds (you can export AWS_PROFILE='ml-admin' right after)
        tar_command = _build_tar_cmd(context, ignore_file, folders_to_include)
        print(f"Creating archive:   {' '.join(tar_command)}")

        with subprocess.Popen(
            tar_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as proc:
            assert proc.stdout is not None
            with storage_client.open(
                s3_uri,
                "wb",
            ) as out_file:
                shutil.copyfileobj(proc.stdout, out_file)
        print("Done uploading!")
    except (ClientError, ProfileNotFound):
        print("Did you gimme_okta_aws_creds and then export AWS_PROFILE='ml-admin'? Try doing both")
        raise


def _build_tar_cmd(
    context: str, ignore_file: Optional[str], folders_to_include: List[str]
) -> List[str]:
    assert len(folders_to_include) > 0, "Need at least one folder to create a tar archive from!"

    tar_command = ["tar", "-C", context]

    if ignore_file is not None:
        ignore_file = os.path.join(context, ignore_file)
        if not os.path.isfile(ignore_file):
            print(
                f"WARNING: File {ignore_file} does not exist in calling context, not using any file as a .dockerignore"
            )
        else:
            tar_command.append("--exclude-from")
            tar_command.append(ignore_file)

    tar_command.append("-cf")
    tar_command.append("-")
    tar_command.extend(folders_to_include)

    return tar_command


def start_build_job(
    s3_file_name: str,
    path_to_dockerfile: str,
    repotags: Iterable[str],
    use_cache: bool,
    cache_name: str,
    build_args: Optional[Dict[str, str]] = None,
    custom_tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    Starts k8s job that builds your docker image
    Need to authenticate as okta-ml-user for this to work

    :param s3_file_name: Bucketfile for context tar.gz, will download from here
    :param path_to_dockerfile: Path to dockerfile in the context provided, e.g. box_detection/deployment/Dockerfile
    :param repotags: Iterable of strings for ECR repo + tag, in the format repo:tag, e.g.
        ("box-trainer:82376418723647",) or ["box-trainer:82376418723647", "box-trainer:latest"],
    :param use_cache: Whether to use repo in ECR as cache for docker build
    :return: Name of k8s job started
    """
    if not custom_tags:
        custom_tags = {}

    custom_tags_serialized = json.dumps(custom_tags)

    destination_template = Template(
        f"--destination={infra_config().docker_repo_prefix}/$REPO_AND_TAG"
    )

    job_name = f"kaniko-{str(uuid.uuid4())[:8]}"
    print(f"Starting job named: {job_name}")
    with ExitStack() as stack:
        f = stack.enter_context(tempfile.NamedTemporaryFile("wt", suffix=".yaml"))
        template_f = stack.enter_context(open(TEMPLATE_FILE, "rt"))

        # In Circle CI we need to retrieve the AWS access key to attach to kaniko
        aws_access_key_id = ""
        aws_secret_access_key = ""
        if os.getenv("CIRCLECI"):
            aws_access_key_id_result = subprocess.run(
                ["aws", "configure", "get", "aws_access_key_id"], check=False, stdout=PIPE
            )
            aws_access_key_id = aws_access_key_id_result.stdout.decode().strip()
            aws_secret_access_key_result = subprocess.run(
                ["aws", "configure", "get", "aws_secret_access_key"], check=False, stdout=PIPE
            )
            aws_secret_access_key = aws_secret_access_key_result.stdout.decode().strip()
        job = Template(template_f.read()).substitute(
            NAME=job_name,
            CUSTOM_TAGS=json.dumps(custom_tags_serialized),
            DOCKERFILE=path_to_dockerfile,
            S3_BUCKET=S3_BUCKET,
            S3_FILE=s3_file_name,
            USE_CACHE="true" if use_cache else "false",
            CACHE_REPO=f"{infra_config().docker_repo_prefix}/{cache_name}",
            AWS_ACCESS_KEY_ID=aws_access_key_id,
            AWS_SECRET_ACCESS_KEY=aws_secret_access_key,
            NAMESPACE=NAMESPACE,
        )
        yml = yaml.safe_load(job)
        destinations = [destination_template.substitute(REPO_AND_TAG=rt) for rt in repotags]
        yml["spec"]["template"]["spec"]["containers"][0]["args"].extend(destinations)

        if build_args:
            yml["spec"]["template"]["spec"]["containers"][0]["args"].extend(
                [f"--build-arg={key}={value}" for key, value in build_args.items()]
            )

        yaml.dump(yml, stream=f, default_flow_style=False)
        f.seek(0)

        container_spec: str = yaml.dump(yml["spec"]["template"]["spec"]["containers"][0]).strip()

        print("Maybe update CodeArtifact token secret")
        if not os.path.exists("/tmp"):
            os.makedirs("/tmp")
        pip_conf_file = "/tmp/.codeartifact-pip-conf"
        aws_profile = infra_config().profile_ml_worker
        try:
            # nosemgrep
            subprocess.check_output(
                [
                    f"AWS_PROFILE={aws_profile} python scripts_py3/scale_scripts/exe/maybe_refresh_codeartifact.py --export {pip_conf_file}"
                ],
                cwd=str(MODELS_ROOT),
                shell=True,
            )
            with open(pip_conf_file) as f_conf:
                pip_conf_data = f_conf.read()
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("WARNING: Failed to refresh CodeArtifact token secret, using empty secret")
            pip_conf_data = ""
        pip_conf_base64 = b64encode(pip_conf_data.encode("utf-8")).decode("utf-8")
        data = {"data": {"codeartifact_pip_conf": pip_conf_base64}}
        subprocess.check_output(
            ["kubectl", "patch", "secret", "codeartifact-pip-conf", f"-p={json.dumps(data)}"]
        ).decode("utf-8")

        print(f"Executing Kaniko build command:\n{container_spec}")
        print("-" * 80)

        print(subprocess.check_output(["kubectl", "apply", "-f", f.name]).decode("utf-8"))
    return job_name


def build_remote(
    context: str,
    dockerfile: str,
    repotags: Union[str, Iterable[str]],
    folders_to_include: Optional[List[str]] = None,
    use_cache: bool = True,
    cache_name: str = "kaniko-cache",
    ignore_file: Optional[str] = None,
    build_args: Optional[Dict[str, str]] = None,
    custom_tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    Entrypoint for building docker images on cluster

    :param context: Build context relative to current directory
    :param dockerfile: Path to dockerfile in the context provided, e.g. box_detection/deployment/Dockerfile
    :param repotags: Iterable of strings for ECR repo + tag, in the format repo:tag, e.g.
        ("box-trainer:82376418723647",) or ["box-trainer:82376418723647", "box-trainer:latest"],
    :param folders_to_include: List of paths to subfolders needed to build docker image relative to context
    :param use_cache: Whether to use repo in ECR as cache for docker build
    :param ignore_file: File (e.g. .dockerignore) containing things to ignore when preparing docker context. Relative to context
    :return: Name of k8s job started
    """

    # Ensures that:
    #   (1) dockerfile exists within context
    #   (2) makes sure this is a filepath relative **from** the context directory
    dockerfile = verify_and_reformat_as_relative_to(context, dockerfile)
    # This new relative filepath will always start with ./ because,
    # during remote building, the build **starts** at the context.

    if isinstance(repotags, str):
        repotags = [repotags]

    # Figure out default folders to include
    calling_path = Path(context).resolve()
    if folders_to_include is None:
        if calling_path == MODELS_ROOT:
            default_folders = {}

            # find the models/ project folder that this Dockerfile comes from
            parts = dockerfile.split("/")
            i = 0
            for i, p in enumerate(parts):
                if p == "models":
                    break
            else:
                raise ValueError(
                    "Cannot figure out the models/ directory where this dockerfile lives! "
                    "Be explicit and pass in `folders_to_include`!"
                )
            project_dir_containing_dockerfile = parts[i + 1]
            default_folders.add(f"{project_dir_containing_dockerfile}/")

            folders_to_include = list(default_folders)
        else:
            folders_to_include = ["."]
    print(f"Using context:      {calling_path}")
    print(f"Including folders:  {folders_to_include}")

    file_uuid = uuid.uuid4()
    s3_file_name = f"{SUB_BUCKET}/{file_uuid}.tar.gz"
    zip_context(
        s3_file_name,
        context=context,
        folders_to_include=folders_to_include,
        ignore_file=ignore_file,
    )
    return start_build_job(
        s3_file_name, dockerfile, repotags, use_cache, cache_name, build_args, custom_tags
    )


def verify_and_reformat_as_relative_to(context: str, dockerfile: str) -> str:
    """Validate that the Dockerfile exists within the context and output a relative path for the Dockerfile.

    This function raises a :raises:`ValueError` if:
        - :param:`context` is not a valid directory
        - the :param:`dockerfile` is not a valid file
        - the :param:`dockerfile`, exists, but is not within the directory tree from :param:`context`

    Otherwise, the function will return a string. This string is the relative filepath to
    :param:`dockerfile` from the :param:`context` directory. Additionally, if the function
    returns a string, you are guaranteed that the above three points are not true (i.e. those
    failing conditions are _not_ met).
    """
    context_p = Path(context).resolve().absolute()
    if not context_p.is_dir():
        raise ValueError(f"{context=} is not a valid directory")

    dockerfile_p = Path(dockerfile).resolve().absolute()
    if not dockerfile_p.is_file():
        dockerfile_within_c = (context_p / dockerfile).resolve()
        if not dockerfile_within_c.is_file():
            raise ValueError(
                f"{dockerfile=} is not a valid file and {context}/{dockerfile} doesn't exist"
            )
        dockerfile_p = dockerfile_within_c

    try:
        dockerfile_relative_to_context = str(dockerfile_p.relative_to(context_p))
    except ValueError:
        logger.exception(f"Dockerfile ({dockerfile}) is not contained within context ({context})")
        raise
    else:
        return f"./{dockerfile_relative_to_context}"


def _read_pod_logs(pod_name):
    return subprocess.check_output(["kubectl", "logs", pod_name, "-n", NAMESPACE, "kaniko"]).decode(
        "utf-8"
    )


def get_pod_status_and_log(job_name: str) -> BuildResult:
    """
    Waits for pod to succeed/fail or for timeout to happen,
    also spawns a subprocess to print the logs of the pod
    Also assumes that the "app" label for pods of the job == job_name

    Should only use it if your job uses exactly one pod ever (e.g. it shouldn't auto-restart failed pods)
    :param job_name: Name of k8s job of given pod
    :return: Whether the pod succeeded
    """
    try:
        kube_config.load_incluster_config()
    except ConfigException:
        print("No cluster config found, using local config")
        kube_config.load_kube_config()

    core_api_instance = client.CoreV1Api()

    @tenacity.retry(
        wait=tenacity.wait_fixed(5),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type(ValueError),
        reraise=True,
    )
    def get_pod_name():
        pods = core_api_instance.list_namespaced_pod(
            NAMESPACE, label_selector=f"app={job_name}"
        ).items
        if len(pods) != 1:
            raise ValueError("No pod created")
        return pods[0].metadata.name

    pod_name = get_pod_name()

    watcher = watch.Watch()
    # There technically is a race between Kaniko finishing its build and the watcher.stream starting
    # but Kaniko shouldn't take less than 30 seconds to build which should be plenty of time
    # Also a race between the Kaniko image getting the "Running" status and the watcher starting,
    # would lead to logs not printing, but this probably also isn't really a problem.
    logs_process = None

    def cleanup_logs_process():
        if logs_process is not None:
            if logs_process.stderr is not None:
                logs_process.stderr.flush()
            if logs_process.stdout is not None:
                logs_process.stdout.flush()
            # give the streaming process a moment to output the last log lines
            try:
                logs_process.wait(1)
            except subprocess.TimeoutExpired:
                pass
            logs_process.kill()
        else:
            # If we don't ever see a "Running" event print out the logs anyways
            subprocess.run(["kubectl", "logs", pod_name, "-n", NAMESPACE, "kaniko"], check=True)

    for event in watcher.stream(
        core_api_instance.list_namespaced_pod,
        namespace=NAMESPACE,
        field_selector=f"metadata.name={pod_name}",
        timeout_seconds=TIMEOUT_SECS,
    ):
        print(f"Pod status: {event['object'].status.phase}")

        if event["object"].status.phase == "Running":
            logs_process = subprocess.Popen(  # pylint: disable=consider-using-with
                ["kubectl", "logs", pod_name, "-n", NAMESPACE, "kaniko", "-f"]
            )
        elif event["object"].status.phase == "Succeeded":
            cleanup_logs_process()
            return BuildResult(status=True, logs=_read_pod_logs(pod_name))
        elif event["object"].status.phase == "Failed":
            cleanup_logs_process()
            return BuildResult(status=False, logs=_read_pod_logs(pod_name))
    if logs_process is not None:
        logs_process.kill()
    return BuildResult(status=False, logs=_read_pod_logs(pod_name))


def build_remote_block(
    context: str,
    dockerfile: str,
    repotags: Union[str, Iterable[str]],
    folders_to_include: Optional[List[str]] = None,
    use_cache: bool = True,
    cache_name: str = "kaniko-cache",
    ignore_file: Optional[str] = None,
    build_args: Optional[Dict[str, str]] = None,
    custom_tags: Optional[Dict[str, str]] = None,
) -> BuildResult:
    """
    Other entrypoint for building docker images on cluster
    Blocks until docker image has built/uploaded or error has occurred

    :param context: Build context relative to current directory
    :param dockerfile: Path to dockerfile in the context provided, e.g. box_detection/deployment/Dockerfile
    :param repotags: Iterable of strings for ECR repo + tag, in the format repo:tag, e.g.
        ("box-trainer:82376418723647",) or ["box-trainer:82376418723647", "box-trainer:latest"],
    :param folders_to_include: List of paths to subfolders needed to build docker image, relative to context
    :param use_cache: Whether to use repo in ECR as cache for docker build
    :param ignore_file: File (e.g. .dockerignore) containing things to ignore when preparing docker context. Relative to context
    :return: BuildResult representing if docker image has successfully built/pushed
    """
    logger.info(f"build_remote_block args {locals()}")
    job_name = build_remote(
        context,
        dockerfile,
        repotags,
        folders_to_include,
        use_cache,
        cache_name,
        ignore_file,
        build_args,
        custom_tags,
    )
    logger.info(f"Waiting for job {job_name} to finish")
    result = get_pod_status_and_log(job_name)
    return result


@click.command()
@click.option(
    "--context",
    required=False,
    default=".",
    help="Build context relative to current directory",
)
@click.option(
    "-f",
    "--dockerfile",
    required=False,
    default="Dockerfile",
    help="Path to dockerfile, relative to context.",
)
@click.option(
    "-t",
    "--repotag",
    required=True,
    multiple=True,
    help="Repo and tag in your standard repo:tag format",
)
@click.option(
    "--folders",
    required=False,
    help="Comma separated list of folders (relative to context",
)
@click.option(
    "--no-cache",
    required=False,
    is_flag=True,
    help="Don't use cache when building docker images",
)
@click.option(
    "--no-block",
    is_flag=True,
    help="Omit any sort of log streaming/monitoring after build/push job started",
)
@click.option(
    "--dockerignore",
    required=False,
    default=".dockerignore",
    help="Path to .dockerignore file, relative to context",
)
@click.option(
    "--build-arg",
    required=False,
    default=None,
    multiple=True,
    help="Dockerfile build args. Can be repeated.",
)
@click.option(
    "--custom-tags",
    required=False,
    default=None,
    help="Custom datadog tags for kaniko build jobs. Should be a serialize dict of the form: "
    '\'{"<TAG_KEY>": "<TAG_VALUE>","<TAG_KEY_1>": "<TAG_VALUE_1>"}\'',
)
def build_remote_wrapper(
    context: str,
    dockerfile: str,
    repotag: Iterable[str],
    folders: Optional[str],
    no_cache: bool,
    no_block: bool,
    dockerignore: str,
    build_arg: Optional[List[str]] = None,
    custom_tags: Optional[str] = None,
):
    """
    Build/push docker images remotely
    See README for further explanation
    """
    custom_tags = json.loads(custom_tags)
    folders_to_include: Optional[List[str]] = folders.split(",") if folders is not None else None

    cache_name = "kaniko-cache"

    build_args = None
    if build_arg:
        build_arg_kvs = [arg.split("=") for arg in build_arg]
        build_args = {k: v for k, v in build_arg_kvs}  # pylint:disable=unnecessary-comprehension

    if no_block:
        build_remote(
            context=context,
            dockerfile=dockerfile,
            repotags=repotag,
            folders_to_include=folders_to_include,
            use_cache=not no_cache,
            cache_name=cache_name,
            ignore_file=dockerignore,
            build_args=build_args,
            custom_tags=custom_tags,
        )
    else:
        build_result = build_remote_block(
            context=context,
            dockerfile=dockerfile,
            repotags=repotag,
            folders_to_include=folders_to_include,
            use_cache=not no_cache,
            cache_name=cache_name,
            ignore_file=dockerignore,
            build_args=build_args,
            custom_tags=custom_tags,
        )
        if not build_result.status:
            raise Exception("Build/push failed, throwing error")
