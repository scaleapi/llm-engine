import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from typing import List, Optional
from urllib.parse import urlparse

import boto3
import smart_open
from func_timeout import FunctionTimedOut, func_set_timeout
from model_engine_server.common.dtos.llms import (
    CompletionOutput,
    CreateBatchCompletionsRequest,
    CreateBatchCompletionsRequestContent,
    TokenOutput,
    ToolConfig,
)
from model_engine_server.inference.tool_completion.tools import TOOL_MAP, BaseTool, Tools, tokenizer
from tqdm import tqdm

CONFIG_FILE = os.getenv("CONFIG_FILE")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

os.environ["AWS_PROFILE"] = os.getenv("S3_WRITE_AWS_PROFILE", "default")


def get_s3_client():
    session = boto3.Session(profile_name=os.getenv("S3_WRITE_AWS_PROFILE"))
    return session.client("s3", region_name=AWS_REGION)


def download_model(checkpoint_path, final_weights_folder):
    s5cmd = f"./s5cmd --numworkers 512 sync --concurrency 10 --include '*.model' --include '*.json' --include '*.bin' --include '*.safetensors' --exclude 'optimizer*' --exclude 'train*' {os.path.join(checkpoint_path, '*')} {final_weights_folder}"
    env = os.environ.copy()
    env["AWS_PROFILE"] = os.getenv("S3_WRITE_AWS_PROFILE", "default")
    # Need to override these env vars so s5cmd uses AWS_PROFILE
    env["AWS_ROLE_ARN"] = ""
    env["AWS_WEB_IDENTITY_TOKEN_FILE"] = ""
    process = subprocess.Popen(
        s5cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )
    for line in process.stdout:
        print(line, flush=True)

    process.wait()

    if process.returncode != 0:
        stderr_lines = []
        for line in iter(process.stderr.readline, ""):
            stderr_lines.append(line.strip())

        print(f"Error downloading model weights: {stderr_lines}", flush=True)


def file_exists(path):
    try:
        with smart_open.open(path, "r"):
            return True
    except Exception as exc:
        print(f"Error checking if file exists: {exc}")
        return False


def parse_s3_url(s3_url):
    parsed_url = urlparse(s3_url)

    if parsed_url.scheme != "s3":
        raise ValueError(f'The URL scheme is not "s3": {s3_url}')

    bucket = parsed_url.netloc
    key = parsed_url.path.lstrip("/")

    return bucket, key


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
    with smart_open.open(request.output_data_path, "w") as f:
        f.write("[")
        for i in range(request.data_parallelism):
            if i > 0:
                f.write(",")
            chunk_file = f"{request.output_data_path}.{i}"
            with smart_open.open(chunk_file, "r") as chunk_f:
                chunk_data = chunk_f.read()
                f.write(chunk_data[1:-1])  # Remove leading and trailing brackets
        f.write("]")
    print("Chunks combined")


def delete_s3_chunks(request):
    print("Deleting S3 chunks...")
    for i in range(request.data_parallelism):
        chunk_file = f"{request.output_data_path}.{i}"
        bucket, key = parse_s3_url(chunk_file)
        get_s3_client().delete_object(Bucket=bucket, Key=key)
    print("Chunks deleted")


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def get_vllm_engine(model, request):
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    engine_args = AsyncEngineArgs(
        model=model,
        quantization=request.model_config.quantize,
        tensor_parallel_size=request.model_config.num_shards,
        seed=request.model_config.seed or 0,
        disable_log_requests=True,
    )

    llm = AsyncLLMEngine.from_engine_args(engine_args)
    return llm


async def generate_with_tool(
    llm,
    tool_config: ToolConfig,
    content: CreateBatchCompletionsRequestContent,
    prompts,
    tool: BaseTool,
):
    class IterativeGeneration:
        def __init__(self, prompt, max_new_tokens):
            self.generated_text = ""
            self.num_prompt_tokens = 0
            self.remaining_tokens = max_new_tokens
            self.token_logits = []
            self.tool_exception = None
            self.prompt = prompt
            self.completed = False

        def __repr__(self) -> str:
            return f"generated_text: {self.generated_text}, num_prompt_tokens: {self.num_prompt_tokens}, remaining_tokens: {self.remaining_tokens}, tool_exception: {self.tool_exception}, prompt: {self.prompt}, completed: {self.completed}"

    num_iters = 0
    generations = [IterativeGeneration(prompt, content.max_new_tokens) for prompt in prompts]

    while num_iters < tool_config.max_iterations:
        num_iters += 1

        iter_prompts = [
            (gen.prompt + gen.generated_text, idx)
            for idx, gen in enumerate(generations)
            if not gen.completed
        ]

        if not iter_prompts:
            break

        bar = tqdm(
            total=len(iter_prompts),
            desc=f"Generating outputs, iteration {num_iters}",
            file=sys.stdout,
        )

        print(f"Before LLM {generations=}")

        outputs = await generate_with_vllm(
            llm,
            content.max_new_tokens,
            content.temperature,
            content.stop_sequences,
            content.return_token_log_probs,
            content.presence_penalty,
            content.frequency_penalty,
            content.top_k,
            content.top_p,
            [iter[0] for iter in iter_prompts],
            bar,
        )

        print(f"After LLM {outputs=}")

        bar = tqdm(
            total=len(iter_prompts),
            desc=f"Running tools, iteration {num_iters}",
            file=sys.stdout,
        )
        for i in range(len(iter_prompts)):
            bar.update(1)
            response = outputs[i]
            gen_item = generations[iter_prompts[i][1]]
            new_text = response.text

            if content.return_token_log_probs:
                gen_item.token_logits += response.tokens

            if not gen_item.num_prompt_tokens:
                gen_item.num_prompt_tokens = response.num_prompt_tokens

            # break the loop if generation is complete even if remaining_tokens>0
            if len(new_text) == 0:
                gen_item.completed = True
                continue

            # To-do write tools to receive response object itself rather than the text
            try:
                # We need to pass the tool/text to a function that times out if the python code can't execute
                @func_set_timeout(tool_config.execution_timeout_sec)
                def tool_func(text: str, past_context: Optional[str]):
                    return tool()(text, past_context)

                past_context = (
                    gen_item.generated_text if tool_config.should_retry_on_error else None
                )
                new_text, num_tool_output_tokens = tool_func(new_text, past_context)

            except (Exception, FunctionTimedOut) as e:
                # If the tool failed, we should add the error message to the generated text and keep going. It should be added right after the
                # tool call token and concluded with the tool_context_end_token.
                new_text = new_text.rsplit(tool.tool_call_token, 1)

                # We can guarantee this because the tool is not called if it doesn't have the tool call token
                # We still want to replace what the LLM thinks the output should be..
                added_text = str(e) + tool.tool_context_end
                subtracted_text = new_text[1]

                new_text = f"{new_text[0]}{tool.tool_call_token}{e}{tool.tool_context_end}"

                # Now let's add the additional tokens
                num_tool_output_tokens = min(
                    len(tokenizer(added_text).input_ids)
                    - len(tokenizer(subtracted_text).input_ids),
                    0,
                )

                # Also, define the tool exception here so we can raise it later
                gen_item.tool_exception = e

            num_completion_tokens = response.num_completion_tokens

            gen_item.remaining_tokens -= num_completion_tokens
            gen_item.remaining_tokens -= num_tool_output_tokens
            gen_item.generated_text += new_text

            # If we didn't just execute a tool, we're done
            if not gen_item.generated_text.endswith(tool.tool_context_end):
                gen_item.completed = True
                continue

        print(f"After tool use {generations=}")

    results = [
        CompletionOutput(
            text=gen_item.generated_text,
            num_prompt_tokens=gen_item.num_prompt_tokens,
            num_completion_tokens=content.max_new_tokens - gen_item.remaining_tokens,
            tokens=gen_item.token_logits if content.return_token_log_probs else None,
        )
        for gen_item in generations
    ]

    return results


async def batch_inference():
    job_index = int(os.getenv("JOB_COMPLETION_INDEX", 0))

    request = CreateBatchCompletionsRequest.parse_file(CONFIG_FILE)

    if request.model_config.checkpoint_path is not None:
        download_model(request.model_config.checkpoint_path, "./model_weights")

    content = request.content
    if content is None:
        with smart_open.open(request.input_data_path, "r") as f:
            content = CreateBatchCompletionsRequestContent.parse_raw(f.read())

    model = (
        "./model_weights" if request.model_config.checkpoint_path else request.model_config.model
    )

    llm = get_vllm_engine(model, request)

    prompts = []
    prompts_per_pod = len(content.prompts) // request.data_parallelism
    if job_index == request.data_parallelism - 1:
        for prompt in content.prompts[prompts_per_pod * job_index :]:
            prompts.append(prompt)
    else:
        for prompt in content.prompts[
            prompts_per_pod * job_index : prompts_per_pod * (job_index + 1)
        ]:
            prompts.append(prompt)

    if request.tool_config is not None:
        tool_enum = Tools(request.tool_config.name)
        tool = TOOL_MAP[tool_enum]
        outputs = await generate_with_tool(llm, request.tool_config, content, prompts, tool)
    else:
        bar = tqdm(total=len(prompts), desc="Processed prompts")

        outputs = await generate_with_vllm(
            llm,
            content.max_new_tokens,
            content.temperature,
            content.stop_sequences,
            content.return_token_log_probs,
            content.presence_penalty,
            content.frequency_penalty,
            content.top_k,
            content.top_p,
            prompts,
            bar,
        )

        bar.close()

    output_dicts = [output.dict() for output in outputs]

    if request.data_parallelism == 1:
        with smart_open.open(request.output_data_path, "w") as f:
            f.write(json.dumps(output_dicts))
    else:
        chunk_file = f"{request.output_data_path}.{job_index}"
        with smart_open.open(chunk_file, "w") as f:
            f.write(json.dumps(output_dicts))
        if job_index == 0:
            wait_for_all_chunks(request)
            combine_all_chunks(request)
            if request.output_data_path.startswith("s3://"):
                delete_s3_chunks(request)


async def generate_with_vllm(
    engine,
    max_new_tokens,
    temperature,
    stop_sequences,
    return_token_log_probs,
    presence_penalty,
    frequency_penalty,
    top_k,
    top_p,
    prompts,
    bar,
) -> List[CompletionOutput]:
    from vllm import SamplingParams
    from vllm.utils import random_uuid

    # Add the requests to the engine.
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        stop=stop_sequences,
        logprobs=1 if return_token_log_probs else None,
        presence_penalty=presence_penalty or 0.0,
        frequency_penalty=frequency_penalty or 0.0,
        top_k=top_k or -1,
        top_p=top_p or 1.0,
    )

    results_generators = []
    for prompt in prompts:
        request_id = random_uuid()
        results_generator = await engine.add_request(
            request_id, prompt, sampling_params, None, time.monotonic()
        )
        results_generators.append(results_generator)

    outputs = []
    for generator in results_generators:
        last_output_text = ""
        tokens = []
        async for request_output in generator:
            if request_output.finished:
                bar.update(1)

            token_text = request_output.outputs[-1].text[len(last_output_text) :]
            log_probs = request_output.outputs[0].logprobs[-1] if return_token_log_probs else None
            last_output_text = request_output.outputs[-1].text

            if return_token_log_probs:
                tokens.append(
                    TokenOutput(
                        token=token_text,
                        log_prob=log_probs[request_output.outputs[0].token_ids[-1]],
                    )
                )

        num_prompt_tokens = len(request_output.prompt_token_ids)
        num_completion_tokens = len(request_output.outputs[0].token_ids)

        output = CompletionOutput(
            text=request_output.outputs[0].text,
            num_prompt_tokens=num_prompt_tokens,
            num_completion_tokens=num_completion_tokens,
        )
        if return_token_log_probs:
            output.tokens = tokens

        outputs.append(output)
    return outputs


if __name__ == "__main__":
    asyncio.run(batch_inference())
