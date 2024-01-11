import asyncio
import json
import os
import subprocess
import time

from model_engine_server.common.dtos.llms import (
    CompletionOutput,
    CreateBatchCompletionsRequest,
    CreateBatchCompletionsRequestContent,
    TokenOutput,
)
from smart_open import open
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.utils import random_uuid

CONFIG_FILE = os.getenv("CONFIG_FILE")
AWS_REGION = "us-west-2"

job_index = int(os.getenv("JOB_COMPLETION_INDEX"), 0)


def download_model(checkpoint_path, final_weights_folder):
    s5cmd = f"./s5cmd --numworkers 512 cp --concurrency 10 {os.path.join(checkpoint_path, '*')} {final_weights_folder}"
    result = subprocess.run(s5cmd, shell=True, capture_output=True, text=True)
    print(result)
    if result.returncode != 0:
        raise Exception(f"Error downloading model weights: {result.stderr}")


def file_exists(path):
    try:
        with open(path, "r"):
            return True
    except FileNotFoundError:
        return False


def wait_for_all_chunks(request):
    # Max wait time is controlled by the batch job timeout
    while True:
        print("Waiting for all chunks to be written...")
        all_chunks_exist = True
        for i in range(request.data_parallelism):
            chunk_file = f"{request.output_data_path}.{i}"
            if not file_exists(chunk_file):
                print(f"Chunk {chunk_file} does not exist yet")
                all_chunks_exist = False
                break
        if all_chunks_exist:
            break
        time.sleep(5)
    print("All chunks written")


def combine_all_chunks(request):
    print("Combining chunks...")
    with open(request.output_data_path, "w") as f:
        f.write("[")
        for i in range(request.data_parallelism):
            if i > 0:
                f.write(",")
            chunk_file = f"{request.output_data_path}.{i}"
            with open(chunk_file, "r") as chunk_f:
                chunk_data = chunk_f.read()
                f.write(chunk_data[1:-1])  # Remove leading and trailing brackets
        f.write("]")
    print("Chunks combined")


async def batch_inference():
    request = CreateBatchCompletionsRequest.parse_file(CONFIG_FILE)

    if request.model_config.checkpoint_path is not None:
        download_model(request.model_config.checkpoint_path, "./model_weights")

    content = request.content
    if content is None:
        with open(request.input_data_path, "r") as f:
            content = CreateBatchCompletionsRequestContent.parse_raw(f.read())

    model = (
        "./model_weights" if request.model_config.checkpoint_path else request.model_config.model
    )

    engine_args = AsyncEngineArgs(
        model=model,
        quantization=request.model_config.quantize,
        tensor_parallel_size=request.model_config.num_shards,
        seed=request.model_config.seed or 0,
    )

    llm = AsyncLLMEngine.from_engine_args(engine_args)

    # Add the requests to the engine.
    sampling_params = SamplingParams(
        max_tokens=content.max_new_tokens,
        temperature=content.temperature,
        stop=content.stop_sequences,
        logprobs=1 if content.return_token_log_probs else None,
        presence_penalty=content.presence_penalty or 0.0,
        frequency_penalty=content.frequency_penalty or 0.0,
        top_k=content.top_k or -1,
        top_p=content.top_p or 1.0,
    )

    results_generators = []
    prompts_per_pod = len(content.prompts) // request.data_parallelism
    for prompt in content.prompts[prompts_per_pod * job_index : prompts_per_pod * (job_index + 1)]:
        request_id = random_uuid()
        results_generator = await llm.add_request(
            request_id, prompt, sampling_params, None, time.monotonic()
        )
        results_generators.append(results_generator)

    outputs = []
    for generator in results_generators:
        last_output_text = ""
        tokens = []
        async for request_output in generator:
            token_text = request_output.outputs[-1].text[len(last_output_text) :]
            log_probs = request_output.outputs[0].logprobs[-1] if sampling_params.logprobs else None
            last_output_text = request_output.outputs[-1].text

            if content.return_token_log_probs:
                tokens.append(TokenOutput(token=token_text, log_prob=list(log_probs.values())[0]))

        num_prompt_tokens = len(request_output.prompt_token_ids)
        num_completion_tokens = len(request_output.outputs[0].token_ids)

        output = CompletionOutput(
            text=request_output.outputs[0].text,
            num_prompt_tokens=num_prompt_tokens,
            num_completion_tokens=num_completion_tokens,
        )
        if content.return_token_log_probs:
            output.tokens = tokens

        outputs.append(output.dict())

    if request.data_parallelism == 1:
        with open(request.output_data_path, "w") as f:
            f.write(json.dumps(outputs))
    else:
        chunk_file = f"{request.output_data_path}.{job_index}"
        with open(chunk_file, "w") as f:
            f.write(json.dumps(outputs))
        if job_index == 0:
            wait_for_all_chunks(request)
            combine_all_chunks(request)


if __name__ == "__main__":
    asyncio.run(batch_inference())
