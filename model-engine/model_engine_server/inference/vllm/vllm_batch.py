# Batch v2

import argparse
import asyncio
import json
import os
import subprocess
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Coroutine,
    Dict,
    List,
    MutableMapping,
    Optional,
    Union,
)

import smart_open
from fastapi import Request
from model_engine_server.common.dtos.llms import (
    BatchCompletionContent,
    BatchCompletionsModelConfig,
    CompletionResponse,
    CompletionV1Output,
    CreateBatchCompletionsEngineRequest,
    CreateBatchCompletionsV1RequestContent,
    TokenOutput,
    VLLMModelConfig,
)
from model_engine_server.inference.infra.gateways.datadog_inference_monitoring_metrics_gateway import (
    DatadogInferenceMonitoringMetricsGateway,
)
from model_engine_server.inference.utils import (
    await_coroutines,
    check_unknown_startup_memory_usage,
    get_cpu_cores_in_container,
    random_uuid,
)
from model_engine_server.inference.vllm.init_ray_batch_inf_v2 import (
    get_node_ip_address,
    init_ray,
    wait_for_head_node_to_exit,
)
from pydantic import TypeAdapter
from starlette.datastructures import Headers
from tqdm import tqdm
from typing_extensions import TypeAlias, assert_never
from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import BaseModelPath
from vllm.utils import merge_async_iterators

CONFIG_FILE = os.getenv("CONFIG_FILE")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
MODEL_WEIGHTS_FOLDER = os.getenv("MODEL_WEIGHTS_FOLDER", "model_weights")

SKIP_AWS_PROFILE_SET = os.getenv("SKIP_AWS_PROFILE_SET", "false").lower() == "true"
if not SKIP_AWS_PROFILE_SET:
    os.environ["AWS_PROFILE"] = os.getenv("S3_WRITE_AWS_PROFILE", "default")


openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion

CPU_COUNT = get_cpu_cores_in_container()

_BatchCompletionContent: TypeAlias = Union[
    CreateBatchCompletionsV1RequestContent,
    List[CompletionRequest],
    List[ChatCompletionRequest],
]


async def dummy_receive() -> MutableMapping[str, Any]:
    return {"type": "continue"}


# jank but create_completion expects a FastAPI Request object
dummy_request = Request(
    scope={
        "type": "http",
        "path": "/",
        "headers": Headers().raw,
        "http_version": "1.1",
        "method": "GET",
        "scheme": "https",
        "client": ("127.0.0.1", 8080),
    },
    # receive fn that doesn't terminate
    receive=dummy_receive,
)


async def download_model(checkpoint_path: str, target_dir: str, trust_remote_code: bool) -> None:
    additional_include = "--include '*.py'" if trust_remote_code else ""
    
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
        aws_cmd = f"aws s3 sync {checkpoint_path.rstrip('/')} {target_dir} --endpoint-url {endpoint_url} --no-progress"
    else:
        aws_cmd = f"aws s3 sync {checkpoint_path.rstrip('/')} {target_dir} --no-progress"
    
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
        env["AWS_EC2_METADATA_DISABLED"] = "true"  # Disable EC2 metadata for GKE (won't affect EKS)
    
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
        
        if process.stdout:
            for line in process.stdout:
                print(line, flush=True)

        process.wait()

        if process.returncode == 0:
            print("Model download completed successfully!", flush=True)
            return
        else:
            # Handle errors
            stderr_lines = []
            if process.stderr:
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


async def generate_v1_completions(
    engine: EngineClient,
    content: CreateBatchCompletionsV1RequestContent,
) -> List[Optional[CompletionV1Output]]:
    prompts = content.prompts
    bar = tqdm(total=len(prompts), desc="Processed prompts")
    sampling_params = SamplingParams(
        max_tokens=content.max_new_tokens,
        temperature=content.temperature,
        stop=content.stop_sequences,
        logprobs=1 if content.return_token_log_probs else None,
        presence_penalty=content.presence_penalty or 0.0,
        frequency_penalty=content.frequency_penalty or 0.0,
        top_k=content.top_k or -1,
        top_p=content.top_p or 1.0,
        skip_special_tokens=(
            content.skip_special_tokens if content.skip_special_tokens is not None else True
        ),
    )

    results_generators: List[AsyncIterator[RequestOutput]] = []
    for prompt in prompts:
        request_id = random_uuid()
        results_generator = engine.generate(
            prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )
        results_generators.append(results_generator)

    return_token_log_probs = True

    generator = merge_async_iterators(*results_generators)
    outputs: List[Optional[CompletionV1Output]] = [None] * len(prompts)
    tokens: List[List[TokenOutput]] = [list() for _ in prompts]
    async for i, res in generator:
        # There should only be one output
        output = res.outputs[-1]

        if return_token_log_probs and output.logprobs is not None:
            # Sometime the logprobs are not present in the output
            logprobs = output.logprobs[-1]
            for token_id in logprobs.keys():
                tokens[i].append(
                    TokenOutput(
                        token=logprobs[token_id].decoded_token,
                        log_prob=logprobs[token_id].logprob,
                    )
                )

        if res.finished:
            outputs[i] = CompletionV1Output(
                text=output.text,
                num_prompt_tokens=len(res.prompt_token_ids),
                num_completion_tokens=len(output.token_ids),
                tokens=[
                    token.model_dump() for token in tokens[i]
                ],  # Not sure why, but pydantic doesn't like when I pass it TokenOutput directly but works when I encode it as a dict...
            )
            bar.update(1)

    return outputs


# This is needed to handle the cases where it takes too long to process all of the requests before
# the configured 'VLLM_ENGINE_ITERATION_TIMEOUT_S' 30s timeout.
def determine_max_concurrent_requests(
    requests: Union[List[CompletionRequest], List[ChatCompletionRequest]]
) -> int:
    # Guided decoding
    # For example, with guided decoding, vLLM initializes a guided decoding logit processor per request, and
    # anecdotally, we're seeing the engine able to handle around 7req/s (for outlines), so set to 30 * 7 ~= 200
    if any(
        request.to_sampling_params(
            default_max_tokens=1, logits_processor_pattern=None
        ).guided_decoding
        for request in requests
    ):
        return 200

    # Kinda arbitrary number
    return 10000


async def generate_v2_completions(
    engine: EngineClient,
    requests: Union[List[CompletionRequest], List[ChatCompletionRequest]],
) -> List[Union[CompletionResponse, ErrorResponse, None]]:
    bar = tqdm(total=len(requests), desc="Processed requests")
    results_generators: List[
        Coroutine[
            Any,
            Any,
            Union[ErrorResponse, AsyncGenerator[str, None], CompletionResponse],
        ]
    ] = []

    max_concurrent_requests = determine_max_concurrent_requests(requests)
    print(f"max_concurrent_requests: {max_concurrent_requests}")
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def process_request(
        request: Union[CompletionRequest, ChatCompletionRequest]
    ) -> Coroutine[
        Any,
        Any,
        Union[ErrorResponse, AsyncGenerator[str, None], CompletionResponse],
    ]:
        async with semaphore:
            if isinstance(request, CompletionRequest):
                return await openai_serving_completion.create_completion(request, dummy_request)
            elif isinstance(request, ChatCompletionRequest):
                return await openai_serving_chat.create_chat_completion(request)
            else:
                assert_never(request)

    for request in requests:
        results_generators.append(process_request(request))

    results_generator = await_coroutines(*results_generators)
    outputs: List[Optional[CompletionResponse]] = [None] * len(requests)

    async for i, res in results_generator:
        if isinstance(res, AsyncGenerator):
            continue
        outputs[i] = res
        bar.update(1)
    return outputs


async def generate_completions(
    engine: EngineClient, request: _BatchCompletionContent
) -> Union[List[Optional[CompletionV1Output]], List[Optional[CompletionResponse]]]:
    if isinstance(request, CreateBatchCompletionsV1RequestContent):
        return await generate_v1_completions(engine, request)
    elif isinstance(request, List):
        return await generate_v2_completions(engine, request)
    else:
        assert_never(request)


async def init_engine(
    model_id: str,
    served_model_name: str,
    request: CreateBatchCompletionsEngineRequest,
) -> EngineClient:
    global openai_serving_chat
    global openai_serving_completion

    if request.attention_backend is not None:
        os.environ["ATTENTION_BACKEND"] = request.attention_backend

    parsed_configs = VLLMModelConfig.model_validate_json(request.model_cfg.model_dump_json())
    if not parsed_configs.max_model_len:
        parsed_configs.max_model_len = request.model_cfg.max_context_length

    print("VLLM additional configs:", parsed_configs.model_dump())

    engine_args_dict = parsed_configs.model_dump(exclude_none=True)
    default_engine_args_dict = dict(
        model=model_id,
        tensor_parallel_size=request.model_cfg.num_shards,
        pipeline_parallel_size=int(
            os.environ.get("NUM_INSTANCES", 1)
        ),  # TODO maybe do something other than TP=8, PP=number of nodes
        seed=request.model_cfg.seed or 0,
        disable_log_requests=True,
        gpu_memory_utilization=request.max_gpu_memory_utilization or 0.9,
    )
    default_engine_args_dict.update(engine_args_dict)

    engine_args = AsyncEngineArgs(**default_engine_args_dict)

    engine_client = AsyncLLMEngine.from_engine_args(engine_args)
    model_config = await engine_client.get_model_config()
    resolved_chat_template = load_chat_template(parsed_configs.chat_template)
    base_model_paths = [BaseModelPath(name=served_model_name, model_path=model_id)]

    openai_serving_chat = OpenAIServingChat(
        engine_client,
        model_config,
        base_model_paths,
        response_role=request.model_cfg.response_role or "assistant",
        lora_modules=None,
        prompt_adapters=None,
        request_logger=None,
        chat_template=resolved_chat_template,
        chat_template_content_format=None,
    )

    openai_serving_completion = OpenAIServingCompletion(
        engine_client,
        model_config,
        base_model_paths,
        lora_modules=None,
        prompt_adapters=None,
        request_logger=None,
    )

    return engine_client


def overwrite_request(request: Dict[str, Any], model: str) -> Dict[str, Any]:
    request["model"] = model
    request["stream"] = False
    return request


def load_batch_content(
    request: CreateBatchCompletionsEngineRequest,
) -> _BatchCompletionContent:
    content = request.content
    if content is None:
        with smart_open.open(request.input_data_path, "r") as f:
            data = json.load(f)
            content = TypeAdapter(BatchCompletionContent).validate_python(data)

    # Recast the content to vLLMs schema
    if isinstance(content, List) and len(content) > 0:
        model = request.model_cfg.model
        return TypeAdapter(
            Union[List[CompletionRequest], List[ChatCompletionRequest]]
        ).validate_python(
            [
                overwrite_request(req.model_dump(exclude_none=True, mode="json"), model)
                for req in content
            ]
        )

    return content


def get_model_id(model_config: BatchCompletionsModelConfig) -> str:
    return MODEL_WEIGHTS_FOLDER if model_config.checkpoint_path else model_config.model


async def handle_batch_job(
    request: CreateBatchCompletionsEngineRequest, multinode: bool, multinode_timeout: int
) -> None:
    metrics_gateway = DatadogInferenceMonitoringMetricsGateway()

    served_model_name = request.model_cfg.model
    model_id = get_model_id(request.model_cfg)

    if request.model_cfg.checkpoint_path:
        await download_model(
            checkpoint_path=request.model_cfg.checkpoint_path,
            target_dir=MODEL_WEIGHTS_FOLDER,
            trust_remote_code=request.model_cfg.trust_remote_code or False,
        )

    if multinode:
        job_completion_index = int(os.environ.get("JOB_COMPLETION_INDEX", 0))
        # Initialize the ray cluster
        leader_addr = os.environ.get("LEADER_ADDR")
        leader_port = os.environ.get("LEADER_PORT")
        num_instances = os.environ.get("NUM_INSTANCES")
        assert (
            leader_addr is not None and leader_port is not None and num_instances is not None
        ), "Leader addr and port and num_instances must be set"

        # Set this so VLLM starts up correctly with the Ray cluster we set up
        os.environ["VLLM_HOST_IP"] = get_node_ip_address(leader_addr)

        # Also necessary for VLLM
        os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
        os.environ["GLOO_SOCKET_IFNAME"] = "eth0"

        # Debug logging
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["VLLM_LOGGING_LEVEL"] = "INFO"

        init_ray(
            leader_addr=leader_addr,
            leader_port=int(leader_port),
            is_leader=job_completion_index == 0,
            cluster_size=int(num_instances),
            timeout=multinode_timeout,
        )

        if job_completion_index > 0:
            # Skip running the batch job code on all but the first node
            await wait_for_head_node_to_exit()
            exit(0)

    content = load_batch_content(request)
    engine = await init_engine(
        model_id,
        served_model_name,
        request=request,
    )

    outputs = await generate_completions(engine, content)
    with smart_open.open(request.output_data_path, "w") as f:
        f.write(json.dumps([output.model_dump() if output else None for output in outputs]))

    metrics_gateway.emit_batch_completions_metric(
        served_model_name,
        use_tool=False,
        num_prompt_tokens=0,
        num_completion_tokens=0,
        is_finetuned=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file-data",
        "--config_file_data",
        type=str,
        default=None,
        help="Optional override for the config file data, as a json string",
    )
    parser.add_argument(
        "--multinode",
        action="store_true",
        default=False,
        help="Whether to run in multinode mode",
    )
    parser.add_argument(
        "--multinode-timeout",
        type=int,
        default=600,
        help="Timeout for multinode mode",
    )
    args = parser.parse_args()

    check_unknown_startup_memory_usage()

    config_file_data = args.config_file_data
    if config_file_data is None:
        if CONFIG_FILE is None or not os.path.exists(CONFIG_FILE):
            raise FileNotFoundError(f"Config file {CONFIG_FILE} not found")
        with open(CONFIG_FILE, "r") as f:
            config_file_data = f.read()

    request = CreateBatchCompletionsEngineRequest.model_validate_json(config_file_data)

    asyncio.run(handle_batch_job(request, args.multinode, args.multinode_timeout))
