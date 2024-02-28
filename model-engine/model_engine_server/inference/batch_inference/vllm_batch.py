import asyncio
import json
import os
import subprocess
import time
from urllib.parse import urlparse

import boto3
import smart_open
from model_engine_server.common.dtos.llms import (
    CompletionOutput,
    CreateBatchCompletionsRequest,
    CreateBatchCompletionsRequestContent,
    TokenOutput,
)
from tqdm import tqdm

CONFIG_FILE = os.getenv("CONFIG_FILE")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

os.environ["AWS_PROFILE"] = os.getenv("S3_WRITE_AWS_PROFILE", "default")


def get_s3_client():
    session = boto3.Session(profile_name=os.getenv("S3_WRITE_AWS_PROFILE"))
    return session.client("s3", region_name=AWS_REGION)


def download_model(checkpoint_path, final_weights_folder):
    s5cmd = f"./s5cmd --numworkers 512 cp --concurrency 10 --include '*.model' --include '*.json' --include '*.bin' --include '*.safetensors' --exclude 'optimizer*' --exclude 'train*' {os.path.join(checkpoint_path, '*')} {final_weights_folder}"
    env = os.environ.copy()
    env["AWS_PROFILE"] = os.getenv("S3_WRITE_AWS_PROFILE", "default")
    # Need to override these env vars so s5cmd uses AWS_PROFILE
    env["AWS_ROLE_ARN"] = ""
    env["AWS_WEB_IDENTITY_TOKEN_FILE"] = ""
    # nosemgrep
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

    results_generators = await generate_with_vllm(request, content, model, job_index)

    bar = tqdm(total=len(results_generators), desc="Processed prompts")

    outputs = []
    for generator in results_generators:
        last_output_text = ""
        tokens = []
        async for request_output in generator:
            if request_output.finished:
                bar.update(1)

            token_text = request_output.outputs[-1].text[len(last_output_text) :]
            log_probs = (
                request_output.outputs[0].logprobs[-1] if content.return_token_log_probs else None
            )
            last_output_text = request_output.outputs[-1].text

            if content.return_token_log_probs:
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
        if content.return_token_log_probs:
            output.tokens = tokens

        outputs.append(output.dict())

    bar.close()

    if request.data_parallelism == 1:
        with smart_open.open(request.output_data_path, "w") as f:
            f.write(json.dumps(outputs))
    else:
        chunk_file = f"{request.output_data_path}.{job_index}"
        with smart_open.open(chunk_file, "w") as f:
            f.write(json.dumps(outputs))
        if job_index == 0:
            wait_for_all_chunks(request)
            combine_all_chunks(request)
            if request.output_data_path.startswith("s3://"):
                delete_s3_chunks(request)


async def generate_with_vllm(request, content, model, job_index):
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.utils import random_uuid

    engine_args = AsyncEngineArgs(
        model=model,
        quantization=request.model_config.quantize,
        tensor_parallel_size=request.model_config.num_shards,
        seed=request.model_config.seed or 0,
        disable_log_requests=True,
        gpu_memory_utilization=0.8,  # To avoid OOM errors when there's host machine GPU usage
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
    return results_generators


def get_gpu_free_memory():
    """Get GPU free memory using nvidia-smi."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        ).decode("utf-8")
        gpu_memory = [int(x) for x in output.strip().split("\n")]
        return gpu_memory
    except subprocess.CalledProcessError:
        return None


def check_unknown_startup_memory_usage():
    """Check for unknown memory usage at startup."""
    gpu_free_memory = get_gpu_free_memory()
    if gpu_free_memory is not None:
        min_mem = min(gpu_free_memory)
        max_mem = max(gpu_free_memory)
        if max_mem - min_mem > 10:
            print(
                f"WARNING: Unbalanced GPU memory usage at start up. This may cause OOM. Memory usage per GPU in MB: {gpu_free_memory}."
            )
            # nosemgrep
            output = subprocess.check_output(["fuser -v /dev/nvidia*"], shell=True).decode("utf-8")
            print(f"Processes using GPU: {output}")


if __name__ == "__main__":
    check_unknown_startup_memory_usage()
    asyncio.run(batch_inference())
