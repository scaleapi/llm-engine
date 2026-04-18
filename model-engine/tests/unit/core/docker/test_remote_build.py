from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from botocore.exceptions import ClientError
from model_engine_server.core.docker import remote_build


def test_read_ignore_patterns_handles_missing_file(tmp_path, capsys):
    patterns = remote_build._read_ignore_patterns(tmp_path, ".dockerignore")

    assert patterns == []
    assert "does not exist" in capsys.readouterr().out


def test_read_ignore_patterns_skips_comments_and_blank_lines(tmp_path):
    ignore_file = tmp_path / ".dockerignore"
    ignore_file.write_text("\n# comment\n./foo\nbar/\n")

    patterns = remote_build._read_ignore_patterns(tmp_path, ".dockerignore")

    assert patterns == ["foo", "bar/"]


def test_normalize_path_for_archive_relative_path(tmp_path):
    folder = tmp_path / "subdir"
    folder.mkdir()

    resolved_path, archive_root = remote_build._normalize_path_for_archive(tmp_path, "subdir")

    assert resolved_path == folder.resolve()
    assert archive_root == "subdir"


def test_normalize_path_for_archive_rejects_path_outside_context(tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="is not contained within context"):
        remote_build._normalize_path_for_archive(tmp_path, str(outside))


@pytest.mark.parametrize(
    ("member_name", "patterns", "should_keep"),
    [
        ("pkg/file.py", ["pkg"], False),
        ("pkg/file.py", ["*.py"], False),
        ("pkg/file.py", ["other"], True),
    ],
)
def test_filter_archive_member(member_name, patterns, should_keep):
    tar_info = mock.Mock()
    tar_info.name = member_name

    result = remote_build._filter_archive_member(tar_info, patterns)

    assert (result is tar_info) is should_keep


def test_zip_context_uploads_filtered_archive(tmp_path):
    context = tmp_path / "context"
    include_dir = context / "pkg"
    include_dir.mkdir(parents=True)
    (include_dir / "keep.txt").write_text("keep")
    (include_dir / "drop.log").write_text("drop")
    (context / ".dockerignore").write_text("*.log\n")

    uploaded = BytesIO()

    class UploadSink:
        def __enter__(self):
            return uploaded

        def __exit__(self, exc_type, exc, tb):
            uploaded.seek(0)
            return False

    with mock.patch.object(remote_build.storage_client, "open", return_value=UploadSink()):
        remote_build.zip_context(
            s3_file_name="bundle.tar.gz",
            context=str(context),
            folders_to_include=["pkg"],
            ignore_file=".dockerignore",
        )

    archive_path = tmp_path / "uploaded.tar.gz"
    archive_path.write_bytes(uploaded.getvalue())
    import tarfile

    with tarfile.open(archive_path, mode="r:gz") as tar:
        names = tar.getnames()

    assert "pkg/keep.txt" in names
    assert "pkg/drop.log" not in names


def test_zip_context_reraises_storage_errors(tmp_path):
    context = tmp_path / "context"
    folder = context / "pkg"
    folder.mkdir(parents=True)
    (folder / "keep.txt").write_text("keep")
    error_response = {"Error": {"Code": "AccessDenied", "Message": "denied"}}

    with mock.patch.object(
        remote_build.storage_client,
        "open",
        side_effect=ClientError(error_response, "PutObject"),
    ):
        with pytest.raises(ClientError):
            remote_build.zip_context(
                s3_file_name="bundle.tar.gz",
                context=str(context),
                folders_to_include=["pkg"],
            )


def test_start_build_job_uses_boto_credentials_for_circleci(tmp_path):
    template_file = tmp_path / "kaniko_template.yaml"
    template_file.write_text(
        """
apiVersion: batch/v1
kind: Job
metadata:
  name: $NAME
spec:
  template:
    spec:
      containers:
        - name: kaniko
          args: []
          env:
            - name: AWS_ACCESS_KEY_ID
              value: "$AWS_ACCESS_KEY_ID"
            - name: AWS_SECRET_ACCESS_KEY
              value: "$AWS_SECRET_ACCESS_KEY"
            - name: AWS_SESSION_TOKEN
              value: "$AWS_SESSION_TOKEN"
"""
    )
    captured = {}

    def fake_check_output(args, cwd=None, shell=False):
        if shell:
            return b""
        if args[:3] == ["kubectl", "patch", "secret"]:
            captured["patch_args"] = args
            return b"patched"
        if args[:3] == ["kubectl", "apply", "-f"]:
            captured["apply_args"] = args
            captured["apply_yaml"] = Path(args[3]).read_text()
            return b"applied"
        raise AssertionError(f"unexpected subprocess call: {args}")

    frozen_credentials = SimpleNamespace(
        access_key="access",
        secret_key="secret",
        token="token",
    )
    credentials = SimpleNamespace(get_frozen_credentials=lambda: frozen_credentials)

    with (
        mock.patch.object(remote_build, "TEMPLATE_FILE", str(template_file)),
        mock.patch.object(
            remote_build,
            "infra_config",
            return_value=SimpleNamespace(
                docker_repo_prefix="repo-prefix",
                profile_ml_worker="default",
            ),
        ),
        mock.patch.dict(remote_build.os.environ, {"CIRCLECI": "true"}, clear=False),
        mock.patch.object(
            remote_build.boto3,
            "Session",
            return_value=mock.Mock(get_credentials=mock.Mock(return_value=credentials)),
        ),
        mock.patch.object(remote_build.subprocess, "check_output", side_effect=fake_check_output),
    ):
        job_name = remote_build.start_build_job(
            s3_file_name="tmp/context.tar.gz",
            path_to_dockerfile="./Dockerfile",
            repotags=["repo/image:tag"],
            use_cache=True,
            cache_name="cache-repo",
            build_args={"ARG1": "VALUE1"},
            custom_tags={"team": "ml"},
        )

    assert job_name.startswith("kaniko-")
    assert captured["patch_args"][:4] == ["kubectl", "patch", "secret", "codeartifact-pip-conf"]
    assert "--destination=repo-prefix/repo/image:tag" in captured["apply_yaml"]
    assert "--build-arg=ARG1=VALUE1" in captured["apply_yaml"]
    assert "name: AWS_ACCESS_KEY_ID" in captured["apply_yaml"]
    assert "value: access" in captured["apply_yaml"]
    assert "name: AWS_SECRET_ACCESS_KEY" in captured["apply_yaml"]
    assert "value: secret" in captured["apply_yaml"]
    assert "name: AWS_SESSION_TOKEN" in captured["apply_yaml"]
    assert "value: token" in captured["apply_yaml"]


def test_build_remote_with_explicit_folders_calls_zip_and_start(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n")

    with (
        mock.patch.object(remote_build, "zip_context") as mock_zip_context,
        mock.patch.object(
            remote_build, "start_build_job", return_value="kaniko-job"
        ) as mock_start_build_job,
    ):
        result = remote_build.build_remote(
            context=str(tmp_path),
            dockerfile=str(dockerfile),
            repotags="repo/image:tag",
            folders_to_include=["model-engine"],
            build_args={"ARG1": "VALUE1"},
        )

    assert result == "kaniko-job"
    mock_zip_context.assert_called_once()
    zip_kwargs = mock_zip_context.call_args.kwargs
    assert zip_kwargs["context"] == str(tmp_path)
    assert zip_kwargs["folders_to_include"] == ["model-engine"]
    mock_start_build_job.assert_called_once()
    start_args = mock_start_build_job.call_args.args
    assert start_args[1] == "./Dockerfile"
    assert start_args[2] == ["repo/image:tag"]
