from unittest import mock

from model_engine_server.common.dtos.docker_repository import BuildImageRequest
from model_engine_server.infra.repositories.ecr_docker_repository import ECRDockerRepository


def test_normalize_build_args_rewrites_only_paths_inside_base(tmp_path):
    base = tmp_path / "repo"
    base.mkdir()
    inside = base / "nested" / "requirements.txt"
    inside.parent.mkdir()
    inside.write_text("x")
    outside = tmp_path / "outside.txt"
    outside.write_text("y")

    normalized = ECRDockerRepository._normalize_build_args(
        str(base),
        {
            "INSIDE": str(inside),
            "OUTSIDE": str(outside),
            "RELATIVE": "already/relative.txt",
            "NON_STRING": 1,
        },
    )

    assert normalized["INSIDE"] == "nested/requirements.txt"
    assert normalized["OUTSIDE"] == str(outside)
    assert normalized["RELATIVE"] == "already/relative.txt"
    assert normalized["NON_STRING"] == 1


def test_normalize_build_args_does_not_rewrite_base_path_itself(tmp_path):
    base = tmp_path / "repo"
    base.mkdir()

    normalized = ECRDockerRepository._normalize_build_args(
        str(base),
        {
            "CONTEXT_ROOT": str(base),
        },
    )

    assert normalized["CONTEXT_ROOT"] == str(base)


def test_build_image_includes_requirements_and_dockerfile_root(tmp_path):
    repo = ECRDockerRepository()
    base = tmp_path / "repo"
    base.mkdir()
    requirements = base / "model-engine" / ".build-context" / "reqs"
    requirements.mkdir(parents=True)
    abs_build_arg = base / "model-engine" / ".build-context" / "reqs" / "requirements.txt"
    abs_build_arg.write_text("x")

    image_request = BuildImageRequest(
        repo="hosted-model-inference/test",
        image_tag="tag",
        aws_profile="default",
        base_path=str(base),
        dockerfile="model-engine/model_engine_server/inference/pytorch_or_tf.user.Dockerfile",
        base_image="python:3.8-slim",
        requirements_folder="model-engine/.build-context/reqs",
        substitution_args={"REQUIREMENTS_FILE": str(abs_build_arg)},
    )

    build_result = mock.Mock(status=True, logs="ok", job_name="job-1")

    with mock.patch(
        "model_engine_server.infra.repositories.ecr_docker_repository.build_remote_block",
        return_value=build_result,
    ) as mock_build_remote_block:
        response = repo.build_image(image_request)

    assert response.status is True
    assert response.logs == "ok"
    assert response.job_name == "job-1"

    mock_build_remote_block.assert_called_once()
    _, kwargs = mock_build_remote_block.call_args
    assert kwargs["folders_to_include"] == [
        "model-engine",
        "model-engine/.build-context/reqs",
    ]
    assert kwargs["build_args"] == {
        "BASE_IMAGE": "python:3.8-slim",
        "REQUIREMENTS_FILE": "model-engine/.build-context/reqs/requirements.txt",
    }


def test_build_image_without_substitution_args_keeps_base_image_only(tmp_path):
    repo = ECRDockerRepository()
    base = tmp_path / "repo"
    base.mkdir()

    image_request = BuildImageRequest(
        repo="hosted-model-inference/test",
        image_tag="tag",
        aws_profile="default",
        base_path=str(base),
        dockerfile="model-engine/Dockerfile",
        base_image="python:3.13-slim",
    )

    build_result = mock.Mock(status=True, logs="ok", job_name="job-2")

    with mock.patch(
        "model_engine_server.infra.repositories.ecr_docker_repository.build_remote_block",
        return_value=build_result,
    ) as mock_build_remote_block:
        response = repo.build_image(image_request)

    assert response.status is True
    _, kwargs = mock_build_remote_block.call_args
    assert kwargs["folders_to_include"] == ["model-engine"]
    assert kwargs["build_args"] == {"BASE_IMAGE": "python:3.13-slim"}
