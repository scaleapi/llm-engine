from model_engine_server.inference.vllm.vllm_startup_wrapper import get_model_dir_from_server_args


def test_get_model_dir_from_server_args():
    assert (
        get_model_dir_from_server_args(
            ["--model", "/mnt/model-cache/model_files", "--port", "5005"]
        )
        == "/mnt/model-cache/model_files"
    )


def test_get_model_dir_from_server_args_falls_back_when_missing_or_malformed():
    assert get_model_dir_from_server_args(["--port", "5005"]) == "model_files"
    assert get_model_dir_from_server_args(["--model"]) == "model_files"
