# Batch v1

import argparse
import asyncio
import json
import multiprocessing
import os
import subprocess
import sys
import time
import uuid
from multiprocessing.pool import ThreadPool
from typing import List, Optional, Type
from urllib.parse import urlparse

import boto3
import smart_open
from func_timeout import FunctionTimedOut, func_set_timeout
from model_engine_server.inference.batch_inference.dto import (
    CompletionOutput,
    CreateBatchCompletionsEngineRequest,
    CreateBatchCompletionsRequestContent,
    TokenOutput,
    ToolConfig,
)
from model_engine_server.inference.infra.gateways.datadog_inference_monitoring_metrics_gateway import (
    DatadogInferenceMonitoringMetricsGateway,
)
from model_engine_server.inference.tool_completion.tools import TOOL_MAP, BaseTool, Tools, tokenizer
from tqdm import tqdm

CONFIG_FILE = os.getenv("CONFIG_FILE")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
MODEL_WEIGHTS_FOLDER = os.getenv("MODEL_WEIGHTS_FOLDER", "./model_weights")

SKIP_AWS_PROFILE_SET = os.getenv("SKIP_AWS_PROFILE_SET", "false").lower() == "true"
if not SKIP_AWS_PROFILE_SET:
    os.environ["AWS_PROFILE"] = os.getenv("S3_WRITE_AWS_PROFILE", "default")


def get_cpu_cores_in_container():
    cpu_count = multiprocessing.cpu_count()
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
            cfs_quota_us = int(fp.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            cfs_period_us = int(fp.read())
        if cfs_quota_us != -1:
            cpu_count = cfs_quota_us // cfs_period_us
    except FileNotFoundError:
        pass
    return cpu_count


CPU_COUNT = get_cpu_cores_in_container()


def get_s3_client():
    from model_engine_server.core.config import infra_config
    
    if infra_config().cloud_provider == "onprem":
        # For onprem, use explicit credentials from environment variables
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=infra_config().default_region
        )
    else:
        session = boto3.Session(profile_name=os.getenv("S3_WRITE_AWS_PROFILE"))
    
    # Support custom endpoints for S3-compatible storage
    endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    client_kwargs = {"region_name": infra_config().default_region}
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    
    return session.client("s3", **client_kwargs)


def download_model(checkpoint_path, final_weights_folder):
    # Support for third-party object storage (like Scality)
    # AWS CLI works with Scality - so we install and use AWS CLI
    endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    
    # Install AWS CLI first (since it's not in the VLLM container by default)
    print("Installing AWS CLI...", flush=True)
    install_process = subprocess.Popen(
        ["pip", "install", "awscli"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    install_process.wait()
    
    if install_process.returncode != 0:
        print("Failed to install AWS CLI", flush=True)
        return
    
    # Simple approach - download all files (basic AWS CLI v1 compatible)
    if endpoint_url:
        aws_cmd = f"aws s3 sync {checkpoint_path.rstrip('/')} {final_weights_folder} --endpoint-url {endpoint_url} --no-progress"
    else:
        aws_cmd = f"aws s3 sync {checkpoint_path.rstrip('/')} {final_weights_folder} --no-progress"
    
    env = os.environ.copy()
    
    # Configure credentials for object storage or AWS
    if endpoint_url:
        # Use object storage credentials when custom endpoint is specified
        env["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "")
        env["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        env["AWS_ENDPOINT_URL"] = endpoint_url
        env["AWS_REGION"] = os.getenv("AWS_REGION", "us-east-1")
        # Disable AWS-specific features for third-party object storage
        env["AWS_EC2_METADATA_DISABLED"] = "true"
        env["AWS_ROLE_ARN"] = ""
        env["AWS_WEB_IDENTITY_TOKEN_FILE"] = ""
    else:
        # Use AWS profile for S3
        env["AWS_PROFILE"] = os.getenv("S3_WRITE_AWS_PROFILE", "default")
        # Need to override these env vars so AWS CLI uses AWS_PROFILE
        env["AWS_ROLE_ARN"] = ""
        env["AWS_WEB_IDENTITY_TOKEN_FILE"] = ""
    
    # Retry logic with exponential backoff
    max_retries = 3
    retry_delay = 10  # seconds
    
    for attempt in range(max_retries):
        print(f"Running AWS CLI command (attempt {attempt + 1}/{max_retries}): {aws_cmd}", flush=True)
        
        process = subprocess.Popen(
            aws_cmd,
            shell=True,  # nosemgrep
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        
        for line in process.stdout:
            print(line, flush=True)

        process.wait()

        if process.returncode == 0:
            print("Model download completed successfully!", flush=True)
            return
        else:
            # Handle errors
            stderr_lines = []
            for line in iter(process.stderr.readline, ""):
                if line.strip():
                    stderr_lines.append(line.strip())

            print(f"Attempt {attempt + 1} failed with return code {process.returncode}", flush=True)
            if stderr_lines:
                print(f"Error output: {stderr_lines}", flush=True)
            
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...", flush=True)
                import time
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"All {max_retries} download attempts failed. Keeping container alive for debugging...", flush=True)
                # Keep container running for debugging instead of raising error
                import time
                while True:
                    print("Container is alive for debugging. Download failed but not exiting.", flush=True)
                    time.sleep(300)  # Print message every 5 minutes 

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


def get_vllm_engine(model: str, request: CreateBatchCompletionsEngineRequest):
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    engine_args = AsyncEngineArgs(
        model=model,
        quantization=request.model_cfg.quantize,
        tensor_parallel_size=request.model_cfg.num_shards,
        seed=request.model_cfg.seed or 0,
        disable_log_requests=True,
        gpu_memory_utilization=request.max_gpu_memory_utilization or 0.9,
        max_model_len=request.max_context_length,
    )

    llm = AsyncLLMEngine.from_engine_args(engine_args)
    return llm


async def generate_with_tool(
    llm,
    tool_config: ToolConfig,
    content: CreateBatchCompletionsRequestContent,
    prompts,
    tool: Type[BaseTool],
    is_finetuned: bool,
    model: str,
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
    max_iterations = tool_config.max_iterations or 10
    stop_sequences = content.stop_sequences or []
    stop_sequences.append(tool.tool_context_end)

    while num_iters < max_iterations:
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

        outputs = await generate_with_vllm(
            llm,
            [generations[iter[1]].remaining_tokens for iter in iter_prompts],
            content.temperature,
            content.stop_sequences,
            content.return_token_log_probs,
            content.presence_penalty,
            content.frequency_penalty,
            content.top_k,
            content.top_p,
            content.skip_special_tokens,
            [iter[0] for iter in iter_prompts],
            bar,
            use_tool=True,
            is_finetuned=is_finetuned,
            model=model,
        )

        bar = tqdm(
            total=len(iter_prompts),
            desc=f"Running tools, iteration {num_iters}",
            file=sys.stdout,
        )

        def tool_func(i):
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
                return

            # To-do write tools to receive response object itself rather than the text
            try:
                # We need to pass the tool/text to a function that times out if the python code can't execute
                @func_set_timeout(tool_config.execution_timeout_seconds)
                def tool_func(text: str, past_context: Optional[str]):
                    return tool()(text, past_context)

                past_context = (
                    gen_item.generated_text if tool_config.should_retry_on_error else None
                )
                new_text, num_tool_output_tokens = tool_func(new_text, past_context)

            except (Exception, FunctionTimedOut) as e:
                # If the tool failed, we should add the error message to the generated text and keep going. It should be added right after the
                # tool call token and concluded with the tool_context_end_token.
                new_text_split = new_text.rsplit(tool.tool_call_token, 1)

                # We can guarantee this because the tool is not called if it doesn't have the tool call token
                # We still want to replace what the LLM thinks the output should be..
                added_text = str(e) + tool.tool_context_end
                subtracted_text = new_text_split[1]

                new_text = f"{new_text_split[0]}{tool.tool_call_token}{e}{tool.tool_context_end}"

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
            if (
                not gen_item.generated_text.endswith(tool.tool_context_end)
                or gen_item.remaining_tokens <= 0
            ):
                gen_item.completed = True

        pool = ThreadPool(CPU_COUNT)
        pool.map(tool_func, range(len(iter_prompts)))

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


async def batch_inference(config_file_data: Optional[str]):
    job_index = int(os.getenv("JOB_COMPLETION_INDEX", 0))

    if config_file_data is None:
        if CONFIG_FILE is None or not os.path.exists(CONFIG_FILE):
            raise FileNotFoundError(f"Config file {CONFIG_FILE} not found")
        with open(CONFIG_FILE, "r") as f:
            config_file_data = f.read()

    request = CreateBatchCompletionsEngineRequest.model_validate_json(config_file_data)

    if request.attention_backend is not None:
        os.environ["VLLM_ATTENTION_BACKEND"] = request.attention_backend

    if request.model_cfg.checkpoint_path is not None:
        download_model(request.model_cfg.checkpoint_path, MODEL_WEIGHTS_FOLDER)

    content = request.content
    if content is None:
        with smart_open.open(request.input_data_path, "r") as f:
            content = CreateBatchCompletionsRequestContent.model_validate_json(f.read())

    model = MODEL_WEIGHTS_FOLDER if request.model_cfg.checkpoint_path else request.model_cfg.model
    is_finetuned = request.model_cfg.checkpoint_path is not None

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
        outputs = await generate_with_tool(
            llm,
            request.tool_config,
            content,
            prompts,
            tool,
            is_finetuned,
            request.model_cfg.model,
        )
    else:
        bar = tqdm(total=len(prompts), desc="Processed prompts")

        outputs = await generate_with_vllm(
            llm,
            [content.max_new_tokens] * len(prompts),
            content.temperature,
            content.stop_sequences,
            content.return_token_log_probs,
            content.presence_penalty,
            content.frequency_penalty,
            content.top_k,
            content.top_p,
            content.skip_special_tokens,
            prompts,
            bar,
            use_tool=False,
            is_finetuned=is_finetuned,
            model=request.model_cfg.model,
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
    skip_special_tokens,
    prompts,
    bar,
    use_tool,
    is_finetuned,
    model,
) -> List[CompletionOutput]:  # pragma: no cover
    from vllm import SamplingParams

    metrics_gateway = DatadogInferenceMonitoringMetricsGateway()

    # Add the requests to the engine.
    results_generators = []
    for idx, prompt in enumerate(prompts):
        request_id = random_uuid()
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens[idx],
            temperature=temperature,
            stop=stop_sequences,
            logprobs=1 if return_token_log_probs else None,
            presence_penalty=presence_penalty or 0.0,
            frequency_penalty=frequency_penalty or 0.0,
            top_k=top_k or -1,
            top_p=top_p or 1.0,
            skip_special_tokens=(skip_special_tokens if skip_special_tokens is not None else True),
        )
        results_generator = await engine.add_request(
            request_id, prompt, sampling_params, time.monotonic(), None
        )
        results_generators.append(results_generator)

    outputs = []
    for generator in results_generators:
        tokens = []
        async for request_output in generator:
            if request_output.finished:
                bar.update(1)

            if return_token_log_probs:
                output = request_output.outputs[0]
                log_probs = output.logprobs[-1] if return_token_log_probs else None
                token_id = output.token_ids[-1]
                tokens.append(
                    TokenOutput(
                        token=log_probs[token_id].decoded_token,
                        log_prob=log_probs[token_id].logprob,
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

        metrics_gateway.emit_batch_completions_metric(
            model, use_tool, num_prompt_tokens, num_completion_tokens, is_finetuned
        )

        outputs.append(output)
    return outputs


def get_gpu_free_memory():  # pragma: no cover
    """Get GPU free memory using nvidia-smi."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout
        gpu_memory = [int(x) for x in output.strip().split("\n")]
        return gpu_memory
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return None


def check_unknown_startup_memory_usage():  # pragma: no cover
    """Check for unknown memory usage at startup."""
    gpu_free_memory = get_gpu_free_memory()
    if gpu_free_memory is not None:
        print(f"GPU free memory at startup in MB: {gpu_free_memory}")
        min_mem = min(gpu_free_memory)
        max_mem = max(gpu_free_memory)
        if max_mem - min_mem > 10:
            print(
                f"WARNING: Unbalanced GPU memory usage at start up. This may cause OOM. Memory usage per GPU in MB: {gpu_free_memory}."
            )
            try:
                output = subprocess.run(
                    ["fuser -v /dev/nvidia*"],
                    shell=True,  # nosemgrep
                    capture_output=True,
                    text=True,
                ).stdout
                print(f"Processes using GPU: {output}")
            except Exception as e:
                print(f"Error getting processes using GPU: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file-data",
        "--config_file_data",
        type=str,
        default=None,
        help="Optional override for the config file data, as a json string",
    )
    args = parser.parse_args()

    check_unknown_startup_memory_usage()
    asyncio.run(batch_inference(args.config_file_data))
