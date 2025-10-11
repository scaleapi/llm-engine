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
    get_node_fqdn,
    get_node_ip_address,
    init_ray,
    wait_for_head_node_to_exit,
)
from pydantic import TypeAdapter
from starlette.datastructures import Headers, State
from tqdm import tqdm
from typing_extensions import TypeAlias, assert_never
from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client,
    build_async_engine_client_from_engine_args,
    init_app_state,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
    ResponsesRequest,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.entrypoints.openai.serving_responses import OpenAIServingResponses
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, merge_async_iterators
from vllm.v1.engine.async_llm import AsyncLLM

CONFIG_FILE = os.getenv("CONFIG_FILE")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
MODEL_WEIGHTS_FOLDER = os.getenv("MODEL_WEIGHTS_FOLDER", "model_weights")

SKIP_AWS_PROFILE_SET = os.getenv("SKIP_AWS_PROFILE_SET", "false").lower() == "true"
if not SKIP_AWS_PROFILE_SET:
    os.environ["AWS_PROFILE"] = os.getenv("S3_WRITE_AWS_PROFILE", "default")

SKIP_MODEL_DOWNLOAD = os.getenv("SKIP_MODEL_DOWNLOAD", "false").lower() == "true"


openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion
openai_serving_responses: OpenAIServingResponses

CPU_COUNT = get_cpu_cores_in_container()

_BatchCompletionContent: TypeAlias = Union[
    CreateBatchCompletionsV1RequestContent,
    List[CompletionRequest],
    List[ChatCompletionRequest],
    List[ResponsesRequest],
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
    import threading

    print(f"Downloading model from {checkpoint_path} to {target_dir}")
    additional_include = "--include '*.py'" if trust_remote_code else ""
    s5cmd = f"./s5cmd --numworkers 512 sync --concurrency 10 --include '*.model' --include '*.json' --include '*.safetensors' --include '*.txt' {additional_include} --exclude 'optimizer*' --exclude 'train*' {os.path.join(checkpoint_path, '*')} {target_dir}"
    print(s5cmd)
    env = os.environ.copy()
    if not SKIP_AWS_PROFILE_SET:
        env["AWS_PROFILE"] = os.getenv("S3_WRITE_AWS_PROFILE", "default")
    print(f"AWS_PROFILE: {env['AWS_PROFILE']}")
    # Need to override these env vars so s5cmd uses AWS_PROFILE
    env["AWS_ROLE_ARN"] = ""
    env["AWS_WEB_IDENTITY_TOKEN_FILE"] = ""
    env["AWS_EC2_METADATA_DISABLED"] = "true"  # Disable EC2 metadata for GKE (won't affect EKS)
    process = subprocess.Popen(
        s5cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    def pump(stream, prefix=""):
        for line in iter(stream.readline, ""):
            print(f"{prefix}{line}", end="", flush=True)
        stream.close()

    t1 = threading.Thread(target=pump, args=(process.stdout, ""))
    t2 = threading.Thread(target=pump, args=(process.stderr, "[ERR] "))
    t1.start()
    t2.start()
    rc = process.wait()
    t1.join()
    t2.join()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, s5cmd)
        print(f"Error downloading model weights: {process.stderr.read()}", flush=True)


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
    requests: Union[List[CompletionRequest], List[ChatCompletionRequest], List[ResponsesRequest]],
) -> int:
    # Guided decoding
    # For example, with guided decoding, vLLM initializes a guided decoding logit processor per request, and
    # anecdotally, we're seeing the engine able to handle around 7req/s (for outlines), so set to 30 * 7 ~= 200
    if any(
        request.to_sampling_params(
            max_tokens=1, logits_processor_pattern=None, default_sampling_params={}
        ).guided_decoding
        for request in requests
        if hasattr(request, 'to_sampling_params')
    ):
        return 200

    # Kinda arbitrary number
    return 10000


async def generate_v2_completions(
    engine: EngineClient,
    requests: Union[List[CompletionRequest], List[ChatCompletionRequest], List[ResponsesRequest]],
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
        request: Union[CompletionRequest, ChatCompletionRequest, ResponsesRequest],
    ) -> Coroutine[
        Any,
        Any,
        Union[ErrorResponse, AsyncGenerator[str, None], CompletionResponse],
    ]:
        async with semaphore:
            if isinstance(request, CompletionRequest):
                return await openai_serving_completion.create_completion(request)
            elif isinstance(request, ChatCompletionRequest):
                return await openai_serving_chat.create_chat_completion(request)
            elif isinstance(request, ResponsesRequest):
                return await openai_serving_responses.create_responses(request)
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
    global openai_serving_responses

    os.environ["VLLM_USE_V1"] = "1"
    if request.attention_backend is not None:
        os.environ["ATTENTION_BACKEND"] = request.attention_backend

    parsed_configs = VLLMModelConfig.model_validate_json(request.model_cfg.model_dump_json())
    if not parsed_configs.max_model_len:
        parsed_configs.max_model_len = request.model_cfg.max_context_length

    print("VLLM additional configs:", parsed_configs.model_dump(), flush=True)

    default_engine_args_dict = dict(
        model=model_id,
        served_model_name=[served_model_name, model_id],
        tensor_parallel_size=request.model_cfg.num_shards,
        pipeline_parallel_size=int(
            os.environ.get("NUM_INSTANCES", 1)
        ),  # TODO maybe do something other than TP=8, PP=number of nodes
        seed=request.model_cfg.seed or 0,
        gpu_memory_utilization=request.max_gpu_memory_utilization or 0.9,
    )
    engine_args_dict = {**default_engine_args_dict, **parsed_configs.model_dump(exclude_none=True)}
    engine_args = AsyncEngineArgs(**engine_args_dict)

    # init engine client
    print("Initializing engine client", flush=True)
    state = State()
    vllm_config = engine_args.create_engine_config(usage_context=UsageContext.OPENAI_BATCH_RUNNER)
    engine_client = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_BATCH_RUNNER,
        enable_log_requests=engine_args.enable_log_requests,
        disable_log_stats=engine_args.disable_log_stats,
        client_addresses=None,
        client_count=1,
        client_index=0)
    await engine_client.reset_mm_cache()

    print("Initialized engine client", flush=True)
    # Scaffolding to set up openai helpers
    default_app_state_args = dict(
        enable_log_requests=False,
        max_log_len=None,
        disable_log_stats=False,
        tool_server=None,
        chat_template_content_format="auto",
        return_tokens_as_token_ids=False,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        structured_outputs_config=argparse.Namespace(reasoning_parser=None),
        enable_prompt_tokens_details=False,
        enable_force_include_usage=False,
        enable_log_outputs=False,
        log_error_stack=False,
        trust_request_chat_template=False,
        exclude_tools_when_tool_choice_none=False,
        enable_server_load_tracking=False,
        chat_template=parsed_configs.chat_template,
        response_role=request.model_cfg.response_role or "assistant",
        lora_modules=None,
    )

    app_state_args = argparse.Namespace(**{
        **default_app_state_args,
        **engine_args_dict,
    })

    await init_app_state(engine_client, vllm_config, state, app_state_args)
    openai_serving_chat = state.openai_serving_chat
    openai_serving_completion = state.openai_serving_completion
    openai_serving_responses = state.openai_serving_responses

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
            Union[List[CompletionRequest], List[ChatCompletionRequest], List[ResponsesRequest]]
        ).validate_python(
            [
                overwrite_request(req.model_dump(exclude_none=True, mode="json"), model)
                for req in content
            ]
        )

    return content


def get_model_id(model_config: BatchCompletionsModelConfig) -> str:
    return MODEL_WEIGHTS_FOLDER if model_config.checkpoint_path else model_config.model


def init_vllm(model_id: str, served_model_name: str, request: CreateBatchCompletionsEngineRequest):

    parsed_configs = VLLMModelConfig.model_validate_json(request.model_cfg.model_dump_json())
    if not parsed_configs.max_model_len:
        parsed_configs.max_model_len = request.model_cfg.max_context_length

    print("VLLM additional configs:", parsed_configs.model_dump(), flush=True)

    default_engine_args_dict = dict(
        served_model_name=[served_model_name, model_id],
        tensor_parallel_size=request.model_cfg.num_shards,
        pipeline_parallel_size=int(
            os.environ.get("NUM_INSTANCES", 1)
        ),
        seed=request.model_cfg.seed or 0,
        gpu_memory_utilization=request.max_gpu_memory_utilization or 0.9,
    )
    engine_args_dict = {**default_engine_args_dict, **parsed_configs.model_dump(exclude_none=True)}

    # convert engine_args_dict to kebab-case --{key}, value pairs to pass into vllm serve subprocess
    # make sure 
    #  * boolean values are passed as --{key} if true, omit if false
    #  * None values are not passed
    #  * list values are passed as --{key}, value1, value2, ...
    vllm_serve_args = []
    for key, value in engine_args_dict.items():
        if value is None:
            continue
        elif isinstance(value, bool):
            if value is True:
                vllm_serve_args.append(f"--{key.replace('_', '-')}")
        elif isinstance(value, list):
            vllm_serve_args.append(f"--{key.replace('_', '-')}")
            for v in value:
                vllm_serve_args.append(str(v))
        else:
            vllm_serve_args.append(f"--{key.replace('_', '-')}")
            vllm_serve_args.append(str(value))

    args = [
        "vllm",
        "serve",
        model_id,
        "--port",
        "8000",
        *vllm_serve_args,
    ]
    print(f"Starting vLLM: {' '.join(args)}", flush=True)
    result = subprocess.run(args)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start vLLM: {result.stderr}")
    print(f"Started vLLM: {result.stdout}", flush=True)


async def handle_batch_job(
    request: CreateBatchCompletionsEngineRequest, multinode: bool, multinode_timeout: int
) -> None:
    metrics_gateway = DatadogInferenceMonitoringMetricsGateway()

    served_model_name = request.model_cfg.model
    model_id = get_model_id(request.model_cfg)
    print(f"Model ID: {model_id}")

    if request.model_cfg.checkpoint_path and not SKIP_MODEL_DOWNLOAD:
        await download_model(
            checkpoint_path=request.model_cfg.checkpoint_path,
            target_dir=MODEL_WEIGHTS_FOLDER,
            trust_remote_code=request.model_cfg.trust_remote_code or False,
        )
        print(f"Downloaded model to {MODEL_WEIGHTS_FOLDER}")

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
        os.environ["VLLM_HOST_IP"] = get_node_ip_address(get_node_fqdn(leader_addr))

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

    # init vllm
    # init_vllm(served_model_name, request)
    # import ray
    # print("[ray] Checking available resources", ray.available_resources())
    # print("[ray] Checking cluster resources", ray.cluster_resources())

    subprocess.run([
        "python", "vllm_batch.py", "--mode", "serve", "--config-file-data", request.model_dump_json()
    ])

    # engine = await init_engine(
    #     model_id,
    #     served_model_name,
    #     request=request,
    # )

    # outputs = await generate_completions(engine, content)
    # with smart_open.open(request.output_data_path, "w") as f:
    #     f.write(json.dumps([output.model_dump() if output else None for output in outputs]))

    # metrics_gateway.emit_batch_completions_metric(
    #     served_model_name,
    #     use_tool=False,
    #     num_prompt_tokens=0,
    #     num_completion_tokens=0,
    #     is_finetuned=True,
    # )

    # engine.shutdown()


async def handle_serve_job(request: CreateBatchCompletionsEngineRequest):
    model_id = get_model_id(request.model_cfg)
    served_model_name = request.model_cfg.model
    content = load_batch_content(request)
    engine = await init_engine(
        model_id,
        served_model_name,
        request=request,
    )

    outputs = await generate_completions(engine, content)
    with smart_open.open(request.output_data_path, "w") as f:
        f.write(json.dumps([output.model_dump() if output else None for output in outputs]))

    engine.shutdown()

def print_debug_info():
    import ray
    import vllm

    print("ray:", ray.__version__)
    print("vllm:", vllm.__version__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["main", "serve"], default="main"
    )
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

    print_debug_info()

    if args.mode == "serve":
        asyncio.run(handle_serve_job(request))

    else:
        asyncio.run(handle_batch_job(request, args.multinode, args.multinode_timeout))
