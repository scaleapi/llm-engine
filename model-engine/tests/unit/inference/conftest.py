from unittest.mock import MagicMock

import pytest
from model_engine_server.common.dtos.llms import (
    CreateBatchCompletionsModelConfig,
    CreateBatchCompletionsRequest,
    CreateBatchCompletionsRequestContent,
)


@pytest.fixture
def create_batch_completions_request():
    return CreateBatchCompletionsRequest(
        model_config=CreateBatchCompletionsModelConfig(
            checkpoint_path="checkpoint_path", model="model", num_shards=4, seed=123, labels={}
        ),
        data_parallelism=1,
        input_data_path="input_data_path",
        output_data_path="output_data_path",
    )


@pytest.fixture
def create_batch_completions_request_content():
    return CreateBatchCompletionsRequestContent(
        prompts=["prompt1", "prompt2"],
        max_new_tokens=100,
        temperature=0.8,
        return_token_log_probs=True,
    )


@pytest.fixture
def create_vllm_request_outputs():
    mock_vllm_request_output1 = MagicMock()
    mock_vllm_request_output1.outputs = [
        MagicMock(text="text1"),
    ]
    mock_vllm_request_output1.prompt_token_ids = [1, 2, 3]
    mock_vllm_request_output1.outputs[0].token_ids = [4]
    mock_vllm_request_output1.outputs[0].logprobs = [{4: 0.1}]

    mock_vllm_request_output2 = MagicMock()
    mock_vllm_request_output2.outputs = [
        MagicMock(text="text1 text2"),
    ]
    mock_vllm_request_output2.prompt_token_ids = [1, 2, 3]
    mock_vllm_request_output2.outputs[0].token_ids = [4, 5]
    mock_vllm_request_output2.outputs[0].logprobs = [{4: 0.1, 5: 0.2}]

    mock_vllm_request_output3 = MagicMock()
    mock_vllm_request_output3.outputs = [
        MagicMock(text="text1 text2 text3"),
    ]
    mock_vllm_request_output3.prompt_token_ids = [1, 2, 3]
    mock_vllm_request_output3.outputs[0].token_ids = [4, 5, 6]
    mock_vllm_request_output3.outputs[0].logprobs = [{4: 0.1, 5: 0.2, 6: 0.3}]
    return [mock_vllm_request_output1, mock_vllm_request_output2, mock_vllm_request_output3]


@pytest.fixture
def mock_s3_client():
    mock_s3_client = MagicMock()
    mock_s3_client.delete_object.return_value = None
    return mock_s3_client


@pytest.fixture
def mock_process():
    mock_process = MagicMock()
    mock_process.stdout = []
    mock_process.stderr.readline.side_effect = [
        "error",
    ]
    mock_process.returncode = 0
    mock_process.wait.return_value = None
    return mock_process
