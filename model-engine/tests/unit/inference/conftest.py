from unittest.mock import MagicMock

import pytest
from model_engine_server.common.dtos.llms import (
    CompletionOutput,
    CreateBatchCompletionsModelConfig,
    CreateBatchCompletionsRequest,
    CreateBatchCompletionsRequestContent,
    TokenOutput,
    ToolConfig,
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
def create_batch_completions_tool_completion_request():
    return CreateBatchCompletionsRequest(
        model_config=CreateBatchCompletionsModelConfig(
            checkpoint_path="checkpoint_path", model="model", num_shards=4, seed=123, labels={}
        ),
        data_parallelism=1,
        input_data_path="input_data_path",
        output_data_path="output_data_path",
        tool_config=ToolConfig(name="code_evaluator"),
    )


@pytest.fixture
def create_batch_completions_tool_completion_request_content():
    return CreateBatchCompletionsRequestContent(
        prompts=["prompt1"],
        max_new_tokens=100,
        temperature=0.8,
        return_token_log_probs=True,
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
    class Logprob:
        """mock, from https://github.com/vllm-project/vllm/blob/v0.4.1/vllm/sequence.py#L18"""

        def __init__(self, logprob: float):
            self.logprob = logprob

    mock_vllm_request_output1 = MagicMock()
    mock_vllm_request_output1.outputs = [
        MagicMock(text="text1"),
    ]
    mock_vllm_request_output1.prompt_token_ids = [1, 2, 3]
    mock_vllm_request_output1.outputs[0].token_ids = [4]
    mock_vllm_request_output1.outputs[0].logprobs = [{4: Logprob(0.1)}]

    mock_vllm_request_output2 = MagicMock()
    mock_vllm_request_output2.outputs = [
        MagicMock(text="text1 text2"),
    ]
    mock_vllm_request_output2.prompt_token_ids = [1, 2, 3]
    mock_vllm_request_output2.outputs[0].token_ids = [4, 5]
    mock_vllm_request_output2.outputs[0].logprobs = [{4: Logprob(0.1), 5: Logprob(0.2)}]

    mock_vllm_request_output3 = MagicMock()
    mock_vllm_request_output3.outputs = [
        MagicMock(text="text1 text2 text3"),
    ]
    mock_vllm_request_output3.prompt_token_ids = [1, 2, 3]
    mock_vllm_request_output3.outputs[0].token_ids = [4, 5, 6]
    mock_vllm_request_output3.outputs[0].logprobs = [
        {4: Logprob(0.1), 5: Logprob(0.2), 6: Logprob(0.3)}
    ]
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


@pytest.fixture
def mock_completion_output():
    return CompletionOutput(
        text="text1 text2 text3",
        num_prompt_tokens=3,
        num_completion_tokens=3,
        tokens=[
            TokenOutput(token="text1", log_prob=0.1),
            TokenOutput(token=" text2", log_prob=0.2),
            TokenOutput(token=" text3", log_prob=0.3),
        ],
    )


@pytest.fixture
def mock_tool_completion_output():
    return CompletionOutput(
        text="```python\nimport math\nprint(math.sqrt(2))\n```\n1.414...\n",
        num_prompt_tokens=10,
        num_completion_tokens=28,
        tokens=[
            TokenOutput(token="``", log_prob=-0.1980377733707428),
            TokenOutput(token="`", log_prob=-0.0037908137310296297),
            TokenOutput(token="python", log_prob=-0.015637163072824478),
            TokenOutput(token="\n", log_prob=-0.0010788579238578677),
            TokenOutput(token="import", log_prob=-0.04351021721959114),
            TokenOutput(token=" math", log_prob=-0.0021214615553617477),
            TokenOutput(token="\n", log_prob=-0.002169043058529496),
            TokenOutput(token="print", log_prob=-0.06555093079805374),
            TokenOutput(token="(", log_prob=-0.005272886715829372),
            TokenOutput(token="math", log_prob=-0.009995171800255775),
            TokenOutput(token=".", log_prob=-0.0002040654799202457),
            TokenOutput(token="sqrt", log_prob=-0.00886327400803566),
            TokenOutput(token="(", log_prob=-0.0015410225605592132),
            TokenOutput(token="2", log_prob=-0.008573509752750397),
            TokenOutput(token="))", log_prob=-0.010970987379550934),
            TokenOutput(token="\n", log_prob=-0.002175347413867712),
            TokenOutput(token="``", log_prob=-0.01911235973238945),
            TokenOutput(token="`", log_prob=-0.0005327236140146852),
            TokenOutput(token="\n", log_prob=-0.002304519060999155),
            TokenOutput(token="1", log_prob=-0.10852570831775665),
            TokenOutput(token=".", log_prob=-0.007146273739635944),
            TokenOutput(token="4", log_prob=-0.003810290014371276),
            TokenOutput(token="1", log_prob=-0.002774677239358425),
            TokenOutput(token="4", log_prob=-0.16946221888065338),
            TokenOutput(token=".", log_prob=-0.007678280584514141),
            TokenOutput(token=".", log_prob=-0.021146666258573532),
            TokenOutput(token=".", log_prob=-0.3870151937007904),
            TokenOutput(token="\n", log_prob=-0.027081478387117386),
        ],
    )


@pytest.fixture
def mock_tool_completion_output2():
    return CompletionOutput(
        text="Final Answer: 4\n",
        num_prompt_tokens=38,
        num_completion_tokens=6,
        tokens=[
            TokenOutput(token="Final", log_prob=-0.1980377733707428),
            TokenOutput(token=" Answer", log_prob=-0.0037908137310296297),
            TokenOutput(token=":", log_prob=-0.015637163072824478),
            TokenOutput(token=" ", log_prob=-0.0010788579238578677),
            TokenOutput(token="4", log_prob=-0.04351021721959114),
            TokenOutput(token="\n", log_prob=-0.0021214615553617477),
        ],
    )


@pytest.fixture
def mock_run_output():
    value = MagicMock()
    value.stdout = "1.4142135623730951"
    value.check_returncode = MagicMock()
    return value
