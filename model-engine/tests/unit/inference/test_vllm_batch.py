import json
from unittest.mock import MagicMock, call, mock_open, patch

import botocore
import pytest
from model_engine_server.inference.batch_inference.vllm_batch import batch_inference, file_exists


@pytest.mark.asyncio
@patch("model_engine_server.inference.batch_inference.vllm_batch.CreateBatchCompletionsRequest")
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
async def test_batch_inference(
    mock_open_func,
    mock_popen,
    mock_get_s3_client,
    mock_generate_with_vllm,
    mock_create_batch_completions_request_content,
    mock_create_batch_completions_request,
    create_batch_completions_request,
    create_batch_completions_request_content,
    create_vllm_request_outputs,
    mock_s3_client,
    mock_process,
    mock_completion_output,
):
    # Mock the necessary objects and data
    mock_popen.return_value = mock_process
    mock_get_s3_client.return_value = mock_s3_client
    mock_create_batch_completions_request.parse_file.return_value = create_batch_completions_request
    mock_create_batch_completions_request_content.parse_raw.return_value = (
        create_batch_completions_request_content
    )

    mock_results_generator = MagicMock()
    mock_results_generator.__aiter__.return_value = create_vllm_request_outputs

    # Mock the generate_with_vllm function
    mock_generate_with_vllm.return_value = [mock_results_generator]

    # Call the function
    await batch_inference()

    # Assertions
    mock_create_batch_completions_request.parse_file.assert_called_once()
    mock_open_func.assert_has_calls(
        [
            call("input_data_path", "r"),
            call("output_data_path", "w"),
            call().write(json.dumps([mock_completion_output.dict()])),
        ],
        any_order=True,
    )


@pytest.mark.asyncio
@patch("model_engine_server.inference.batch_inference.vllm_batch.CreateBatchCompletionsRequest")
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
async def test_batch_inference_failed_to_download_model(
    mock_open_func,
    mock_popen,
    mock_get_s3_client,
    mock_generate_with_vllm,
    mock_create_batch_completions_request_content,
    mock_create_batch_completions_request,
    create_batch_completions_request,
    create_batch_completions_request_content,
    create_vllm_request_outputs,
    mock_s3_client,
    mock_process,
):
    # Mock the necessary objects and data
    mock_process.returncode = 1
    mock_popen.return_value = mock_process
    mock_get_s3_client.return_value = mock_s3_client
    mock_create_batch_completions_request.parse_file.return_value = create_batch_completions_request
    mock_create_batch_completions_request_content.parse_raw.return_value = (
        create_batch_completions_request_content
    )

    # Call the function
    with pytest.raises(IOError):
        await batch_inference()


@pytest.mark.asyncio
@patch("model_engine_server.inference.batch_inference.vllm_batch.CreateBatchCompletionsRequest")
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
@patch("model_engine_server.inference.batch_inference.vllm_batch.os.getenv")
async def test_batch_inference_two_workers(
    mock_getenv,
    mock_open_func,
    mock_popen,
    mock_get_s3_client,
    mock_generate_with_vllm,
    mock_create_batch_completions_request_content,
    mock_create_batch_completions_request,
    create_batch_completions_request,
    create_batch_completions_request_content,
    create_vllm_request_outputs,
    mock_s3_client,
    mock_process,
    mock_completion_output,
):
    # Mock the necessary objects and data
    mock_popen.return_value = mock_process
    mock_get_s3_client.return_value = mock_s3_client
    create_batch_completions_request.data_parallelism = 2
    mock_create_batch_completions_request.parse_file.return_value = create_batch_completions_request
    mock_create_batch_completions_request_content.parse_raw.return_value = (
        create_batch_completions_request_content
    )

    mock_results_generator = MagicMock()
    mock_results_generator.__aiter__.return_value = create_vllm_request_outputs

    # Mock the generate_with_vllm function
    mock_generate_with_vllm.return_value = [mock_results_generator]

    indexes = [1, 0]

    def side_effect(key, default):
        if key == "JOB_COMPLETION_INDEX":
            return indexes.pop(0)
        return default

    mock_getenv.side_effect = side_effect
    # Batch completion worker 1
    await batch_inference()

    # Assertions
    mock_create_batch_completions_request.parse_file.assert_called_once()
    mock_open_func.assert_has_calls(
        [
            call("input_data_path", "r"),
            call("output_data_path.1", "w"),
            call().write(json.dumps([mock_completion_output.dict()])),
        ],
        any_order=True,
    )

    # Batch completion worker 0
    await batch_inference()
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
@patch("model_engine_server.inference.batch_inference.vllm_batch.CreateBatchCompletionsRequest")
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
@patch("model_engine_server.inference.batch_inference.vllm_batch.os.getenv")
async def test_batch_inference_delete_chunks(
    mock_getenv,
    mock_open_func,
    mock_popen,
    mock_get_s3_client,
    mock_generate_with_vllm,
    mock_create_batch_completions_request_content,
    mock_create_batch_completions_request,
    create_batch_completions_request,
    create_batch_completions_request_content,
    create_vllm_request_outputs,
    mock_s3_client,
    mock_process,
    mock_completion_output,
):
    # Mock the necessary objects and data
    mock_popen.return_value = mock_process
    mock_get_s3_client.return_value = mock_s3_client
    create_batch_completions_request.data_parallelism = 2
    create_batch_completions_request.output_data_path = "s3://bucket/key"
    mock_create_batch_completions_request.parse_file.return_value = create_batch_completions_request
    mock_create_batch_completions_request_content.parse_raw.return_value = (
        create_batch_completions_request_content
    )

    mock_results_generator = MagicMock()
    mock_results_generator.__aiter__.return_value = create_vllm_request_outputs

    # Mock the generate_with_vllm function
    mock_generate_with_vllm.return_value = [mock_results_generator]

    indexes = [1, 0]

    def side_effect(key, default):
        if key == "JOB_COMPLETION_INDEX":
            return indexes.pop(0)
        return default

    mock_getenv.side_effect = side_effect
    # Batch completion worker 1
    await batch_inference()

    # Assertions
    mock_create_batch_completions_request.parse_file.assert_called_once()
    mock_open_func.assert_has_calls(
        [
            call("input_data_path", "r"),
            call("s3://bucket/key.1", "w"),
            call().write(json.dumps([mock_completion_output.dict()])),
        ],
        any_order=True,
    )

    # Batch completion worker 0
    await batch_inference()
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
        "model_engine_server.inference.batch_inference.vllm_batch.smart_open.open", mock_open_func
    ):
        result = file_exists(path)

    mock_open_func.assert_called_once_with(path, "r")
    assert result is True


def test_file_exists_no_such_key():
    path = "test_path"

    with patch(
        "model_engine_server.inference.batch_inference.vllm_batch.smart_open.open",
        side_effect=botocore.errorfactory.NoSuchKey,
    ):
        result = file_exists(path)

    assert result is False
