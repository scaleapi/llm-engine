import json
from unittest.mock import call, mock_open, patch

import pytest
from model_engine_server.inference.batch_inference.vllm_batch import batch_inference, file_exists


@pytest.mark.asyncio
@patch("model_engine_server.inference.batch_inference.vllm_batch.get_vllm_engine")
@patch(
    "model_engine_server.inference.batch_inference.vllm_batch.CreateBatchCompletionsEngineRequest"
)
@patch(
    "model_engine_server.inference.batch_inference.vllm_batch.CreateBatchCompletionsRequestContent"
)
@patch("model_engine_server.inference.batch_inference.vllm_batch.generate_with_vllm")
@patch("model_engine_server.inference.batch_inference.vllm_batch.get_s3_client")
@patch("subprocess.Popen")
@patch(
    "model_engine_server.inference.batch_inference.vllm_batch.smart_open.open",
    new_callable=mock_open,
    read_data="Mocked content",
)
@patch("builtins.open", new_callable=mock_open, read_data="Mocked content")
async def test_batch_inference(
    mock_builtins_open_func,
    mock_open_func,
    mock_popen,
    mock_get_s3_client,
    mock_generate_with_vllm,
    mock_create_batch_completions_request_content,
    mock_create_batch_completions_engine_request,
    mock_vllm,
    create_batch_completions_engine_request,
    create_batch_completions_request_content,
    mock_s3_client,
    mock_process,
    mock_completion_output,
):
    # Mock the necessary objects and data
    mock_popen.return_value = mock_process
    mock_get_s3_client.return_value = mock_s3_client
    mock_create_batch_completions_engine_request.model_validate_json.return_value = (
        create_batch_completions_engine_request
    )
    mock_create_batch_completions_request_content.model_validate_json.return_value = (
        create_batch_completions_request_content
    )

    # Mock the generate_with_vllm function
    mock_generate_with_vllm.return_value = [mock_completion_output]

    # Call the function
    await batch_inference("this config data gets ignored because we mock model_validate_json")

    # Assertions
    mock_create_batch_completions_engine_request.model_validate_json.assert_called_once()
    mock_open_func.assert_has_calls(
        [
            call("input_data_path", "r"),
            call("output_data_path", "w"),
            call().write(json.dumps([mock_completion_output.dict()])),
        ],
        any_order=True,
    )


@pytest.mark.asyncio
@patch("model_engine_server.inference.batch_inference.vllm_batch.get_vllm_engine")
@patch(
    "model_engine_server.inference.batch_inference.vllm_batch.CreateBatchCompletionsEngineRequest"
)
@patch(
    "model_engine_server.inference.batch_inference.vllm_batch.CreateBatchCompletionsRequestContent"
)
@patch("model_engine_server.inference.batch_inference.vllm_batch.generate_with_vllm")
@patch("model_engine_server.inference.batch_inference.vllm_batch.get_s3_client")
@patch("subprocess.Popen")
@patch(
    "model_engine_server.inference.batch_inference.vllm_batch.smart_open.open",
    new_callable=mock_open,
    read_data="Mocked content",
)
@patch("builtins.open", new_callable=mock_open, read_data="Mocked content")
async def test_batch_inference_failed_to_download_model_but_proceed(
    mock_builtins_open_func,
    mock_open_func,
    mock_popen,
    mock_get_s3_client,
    mock_generate_with_vllm,
    mock_create_batch_completions_request_content,
    mock_create_batch_completions_engine_request,
    mock_vllm,
    create_batch_completions_engine_request,
    create_batch_completions_request_content,
    mock_s3_client,
    mock_process,
    mock_completion_output,
):
    # Mock the necessary objects and data
    mock_process.returncode = 1  # Failed to download model
    mock_popen.return_value = mock_process
    mock_get_s3_client.return_value = mock_s3_client
    mock_create_batch_completions_engine_request.model_validate_json.return_value = (
        create_batch_completions_engine_request
    )
    mock_create_batch_completions_request_content.model_validate_json.return_value = (
        create_batch_completions_request_content
    )

    # Mock the generate_with_vllm function
    mock_generate_with_vllm.return_value = [mock_completion_output]

    # Call the function
    await batch_inference("this config data gets ignored because we mock model_validate_json")

    # Assertions
    mock_create_batch_completions_engine_request.model_validate_json.assert_called_once()
    mock_open_func.assert_has_calls(
        [
            call("input_data_path", "r"),
            call("output_data_path", "w"),
            call().write(json.dumps([mock_completion_output.dict()])),
        ],
        any_order=True,
    )


@pytest.mark.asyncio
@patch("model_engine_server.inference.batch_inference.vllm_batch.get_vllm_engine")
@patch(
    "model_engine_server.inference.batch_inference.vllm_batch.CreateBatchCompletionsEngineRequest"
)
@patch(
    "model_engine_server.inference.batch_inference.vllm_batch.CreateBatchCompletionsRequestContent"
)
@patch("model_engine_server.inference.batch_inference.vllm_batch.generate_with_vllm")
@patch("model_engine_server.inference.batch_inference.vllm_batch.get_s3_client")
@patch("subprocess.Popen")
@patch(
    "model_engine_server.inference.batch_inference.vllm_batch.smart_open.open",
    new_callable=mock_open,
    read_data="Mocked content",
)
@patch("builtins.open", new_callable=mock_open, read_data="Mocked content")
@patch("model_engine_server.inference.batch_inference.vllm_batch.os.getenv")
async def test_batch_inference_two_workers(
    mock_getenv,
    mock_builtins_open_func,
    mock_open_func,
    mock_popen,
    mock_get_s3_client,
    mock_generate_with_vllm,
    mock_create_batch_completions_request_content,
    mock_create_batch_completions_engine_request,
    mock_vllm,
    create_batch_completions_engine_request,
    create_batch_completions_request_content,
    mock_s3_client,
    mock_process,
    mock_completion_output,
):
    # Mock the necessary objects and data
    mock_popen.return_value = mock_process
    mock_get_s3_client.return_value = mock_s3_client
    create_batch_completions_engine_request.data_parallelism = 2
    mock_create_batch_completions_engine_request.model_validate_json.return_value = (
        create_batch_completions_engine_request
    )
    mock_create_batch_completions_request_content.model_validate_json.return_value = (
        create_batch_completions_request_content
    )

    # Mock the generate_with_vllm function
    mock_generate_with_vllm.return_value = [mock_completion_output]

    indexes = [1, 0]

    def side_effect(key, default):
        if key == "JOB_COMPLETION_INDEX":
            return indexes.pop(0)
        return default

    mock_getenv.side_effect = side_effect
    # Batch completion worker 1
    await batch_inference("this config data gets ignored because we mock model_validate_json")

    # Assertions
    mock_create_batch_completions_engine_request.model_validate_json.assert_called_once()
    mock_open_func.assert_has_calls(
        [
            call("input_data_path", "r"),
            call("output_data_path.1", "w"),
            call().write(json.dumps([mock_completion_output.dict()])),
        ],
        any_order=True,
    )

    # Batch completion worker 0
    await batch_inference("this config data gets ignored because we mock model_validate_json")
    mock_open_func.assert_has_calls(
        [
            call("input_data_path", "r"),
            call("output_data_path.1", "r"),
            call("output_data_path.0", "w"),
            call("output_data_path.0", "r"),
            call("output_data_path", "w"),
            call().write(json.dumps([mock_completion_output.dict()])),
            call().write("["),
            call().write(","),
            call().write("]"),
        ],
        any_order=True,
    )


@pytest.mark.asyncio
@patch("model_engine_server.inference.batch_inference.vllm_batch.get_vllm_engine")
@patch(
    "model_engine_server.inference.batch_inference.vllm_batch.CreateBatchCompletionsEngineRequest"
)
@patch(
    "model_engine_server.inference.batch_inference.vllm_batch.CreateBatchCompletionsRequestContent"
)
@patch("model_engine_server.inference.batch_inference.vllm_batch.generate_with_vllm")
@patch("model_engine_server.inference.batch_inference.vllm_batch.get_s3_client")
@patch("subprocess.Popen")
@patch(
    "model_engine_server.inference.batch_inference.vllm_batch.smart_open.open",
    new_callable=mock_open,
    read_data="Mocked content",
)
@patch("builtins.open", new_callable=mock_open, read_data="Mocked content")
@patch("model_engine_server.inference.batch_inference.vllm_batch.os.getenv")
async def test_batch_inference_delete_chunks(
    mock_getenv,
    mock_builtins_open_func,
    mock_open_func,
    mock_popen,
    mock_get_s3_client,
    mock_generate_with_vllm,
    mock_create_batch_completions_request_content,
    mock_create_batch_completions_engine_request,
    mock_vllm,
    create_batch_completions_engine_request,
    create_batch_completions_request_content,
    mock_s3_client,
    mock_process,
    mock_completion_output,
):
    # Mock the necessary objects and data
    mock_popen.return_value = mock_process
    mock_get_s3_client.return_value = mock_s3_client
    create_batch_completions_engine_request.data_parallelism = 2
    create_batch_completions_engine_request.output_data_path = "s3://bucket/key"
    mock_create_batch_completions_engine_request.model_validate_json.return_value = (
        create_batch_completions_engine_request
    )
    mock_create_batch_completions_request_content.model_validate_json.return_value = (
        create_batch_completions_request_content
    )

    # Mock the generate_with_vllm function
    mock_generate_with_vllm.return_value = [mock_completion_output]

    indexes = [1, 0]

    def side_effect(key, default):
        if key == "JOB_COMPLETION_INDEX":
            return indexes.pop(0)
        return default

    mock_getenv.side_effect = side_effect
    # Batch completion worker 1
    await batch_inference("this config data gets ignored because we mock model_validate_json")

    # Assertions
    mock_create_batch_completions_engine_request.model_validate_json.assert_called_once()
    mock_open_func.assert_has_calls(
        [
            call("input_data_path", "r"),
            call("s3://bucket/key.1", "w"),
            call().write(json.dumps([mock_completion_output.dict()])),
        ],
        any_order=True,
    )

    # Batch completion worker 0
    await batch_inference("this config data gets ignored because we mock model_validate_json")
    mock_open_func.assert_has_calls(
        [
            call("input_data_path", "r"),
            call("s3://bucket/key.1", "r"),
            call("s3://bucket/key.0", "w"),
            call("s3://bucket/key.0", "r"),
            call("s3://bucket/key", "w"),
            call().write(json.dumps([mock_completion_output.dict()])),
            call().write("["),
            call().write(","),
            call().write("]"),
        ],
        any_order=True,
    )

    mock_s3_client.delete_object.assert_has_calls(
        [call(Bucket="bucket", Key="key.0"), call(Bucket="bucket", Key="key.1")]
    )


def test_file_exists():
    mock_open_func = mock_open()
    path = "test_path"

    with patch(
        "model_engine_server.inference.batch_inference.vllm_batch.smart_open.open",
        mock_open_func,
    ):
        result = file_exists(path)

    mock_open_func.assert_called_once_with(path, "r")
    assert result is True


def test_file_exists_no_such_key():
    path = "test_path"

    with patch(
        "model_engine_server.inference.batch_inference.vllm_batch.smart_open.open",
        side_effect=IOError("No such key"),
    ):
        result = file_exists(path)

    assert result is False


@pytest.mark.asyncio
@patch("model_engine_server.inference.batch_inference.vllm_batch.get_vllm_engine")
@patch(
    "model_engine_server.inference.batch_inference.vllm_batch.CreateBatchCompletionsEngineRequest"
)
@patch(
    "model_engine_server.inference.batch_inference.vllm_batch.CreateBatchCompletionsRequestContent"
)
@patch("model_engine_server.inference.batch_inference.vllm_batch.generate_with_vllm")
@patch("model_engine_server.inference.batch_inference.vllm_batch.get_s3_client")
@patch("model_engine_server.inference.batch_inference.vllm_batch.subprocess.Popen")
@patch("subprocess.run")
@patch(
    "model_engine_server.inference.batch_inference.vllm_batch.smart_open.open",
    new_callable=mock_open,
    read_data="Mocked content",
)
@patch("builtins.open", new_callable=mock_open, read_data="Mocked content")
async def test_batch_inference_tool_completion(
    mock_builtins_open_func,
    mock_open_func,
    mock_run,
    mock_popen,
    mock_get_s3_client,
    mock_generate_with_vllm,
    mock_create_batch_completions_request_content,
    mock_create_batch_completions_engine_request,
    mock_vllm,
    create_batch_completions_tool_completion_request,
    create_batch_completions_tool_completion_request_content,
    mock_s3_client,
    mock_process,
    mock_tool_completion_output,
    mock_tool_completion_output2,
    mock_run_output,
):
    # Mock the necessary objects and data
    mock_run.return_value = mock_run_output
    mock_popen.return_value = mock_process
    mock_get_s3_client.return_value = mock_s3_client
    mock_create_batch_completions_engine_request.model_validate_json.return_value = (
        create_batch_completions_tool_completion_request
    )
    mock_create_batch_completions_request_content.model_validate_json.return_value = (
        create_batch_completions_tool_completion_request_content
    )

    # Mock the generate_with_vllm function
    mock_generate_with_vllm.side_effect = [
        [mock_tool_completion_output],
        [mock_tool_completion_output2],
    ]

    # Call the function
    await batch_inference("this config data gets ignored because we mock model_validate_json")

    # Assertions
    mock_create_batch_completions_engine_request.model_validate_json.assert_called_once()
    mock_open_func.assert_has_calls(
        [
            call("input_data_path", "r"),
            call("output_data_path", "w"),
            call().write(
                json.dumps(
                    [
                        {
                            "text": "```python\nimport math\nprint(math.sqrt(2))\n```\n1.4142135623730951\n>>>\nFinal Answer: 4\n",
                            "num_prompt_tokens": 10,
                            "num_completion_tokens": 49,
                            "tokens": [
                                {"token": "``", "log_prob": -0.1980377733707428},
                                {"token": "`", "log_prob": -0.0037908137310296297},
                                {"token": "python", "log_prob": -0.015637163072824478},
                                {"token": "\n", "log_prob": -0.0010788579238578677},
                                {"token": "import", "log_prob": -0.04351021721959114},
                                {"token": " math", "log_prob": -0.0021214615553617477},
                                {"token": "\n", "log_prob": -0.002169043058529496},
                                {"token": "print", "log_prob": -0.06555093079805374},
                                {"token": "(", "log_prob": -0.005272886715829372},
                                {"token": "math", "log_prob": -0.009995171800255775},
                                {"token": ".", "log_prob": -0.0002040654799202457},
                                {"token": "sqrt", "log_prob": -0.00886327400803566},
                                {"token": "(", "log_prob": -0.0015410225605592132},
                                {"token": "2", "log_prob": -0.008573509752750397},
                                {"token": "))", "log_prob": -0.010970987379550934},
                                {"token": "\n", "log_prob": -0.002175347413867712},
                                {"token": "``", "log_prob": -0.01911235973238945},
                                {"token": "`", "log_prob": -0.0005327236140146852},
                                {"token": "\n", "log_prob": -0.002304519060999155},
                                {"token": "1", "log_prob": -0.10852570831775665},
                                {"token": ".", "log_prob": -0.007146273739635944},
                                {"token": "4", "log_prob": -0.003810290014371276},
                                {"token": "1", "log_prob": -0.002774677239358425},
                                {"token": "4", "log_prob": -0.16946221888065338},
                                {"token": ".", "log_prob": -0.007678280584514141},
                                {"token": ".", "log_prob": -0.021146666258573532},
                                {"token": ".", "log_prob": -0.3870151937007904},
                                {"token": "\n", "log_prob": -0.027081478387117386},
                                {"token": "Final", "log_prob": -0.1980377733707428},
                                {
                                    "token": " Answer",
                                    "log_prob": -0.0037908137310296297,
                                },
                                {"token": ":", "log_prob": -0.015637163072824478},
                                {"token": " ", "log_prob": -0.0010788579238578677},
                                {"token": "4", "log_prob": -0.04351021721959114},
                                {"token": "\n", "log_prob": -0.0021214615553617477},
                            ],
                        }
                    ]
                )
            ),
        ],
        any_order=True,
    )
