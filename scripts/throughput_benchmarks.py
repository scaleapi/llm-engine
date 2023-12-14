import json
import os
import queue
import random
import threading
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import requests
import typer
import yaml
from lorem_text import lorem
from transformers import AutoTokenizer

AUTH_USER_ID = os.getenv("AUTH_USER_ID")
GATEWAY_URL = os.getenv("GATEWAY_URL")
app = typer.Typer(name="throughput-benchmarks", add_completion=False)

MAX_CONTEXT_WINDOW = 4096


@dataclass
class BenchmarkConfig:
    def __init__(self, input_token_count, output_token_count_mean):
        self.input_token_count = input_token_count
        self.output_token_count_mean = output_token_count_mean
        # Here we assume 3x standard deviation is enough to cover the range of output token counts.
        # Also assume 3x stddev is rougly half of the mean.
        self.output_token_count_std = output_token_count_mean / 6.0

    def __repr__(self) -> str:
        return f"BenchmarkConfig(input_token_count={self.input_token_count}, output_token_count_mean={self.output_token_count_mean}, output_token_count_std={self.output_token_count_std})"


HF_MODEL_MAPPING = {
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
}


class InferenceFramework(Enum):
    TEXT_GENERATION_INFERENCE = "tgi"
    VLLM = "vllm"
    LIGHTLLM = "lightllm"
    TENSORRT_LLM = "tensorrt-llm"

    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"No member with value {value} in {cls.__name__}")


def send_request(url, request, user=None):
    start = time.time()
    response = requests.post(
        url,
        json=request,
        auth=(user, ""),
        stream=True,
    )
    first_line = True
    for byte_payload in response.iter_lines():
        if first_line:
            time_to_first_token = time.time() - start
            first_line = False

        # Skip line
        if byte_payload == b"\n":
            continue

        payload = byte_payload.decode("utf-8")

        # Event data
        if payload.startswith("data:"):
            payload_data = payload.lstrip("data:").rstrip("/n")
            payload_json = json.loads(payload_data)

    return {
        "payload": payload_json,
        "time_to_first_token": time_to_first_token,
        "total_time": time.time() - start,
    }


def pull_and_send_request_from_queue(
    model: str,
    request_queue: queue.Queue,
    result_queue: queue.Queue,
    use_localhost: bool,
    framework: InferenceFramework,
    local_port: int = 5005,
):
    while not request_queue.empty():
        request = request_queue.get()
        if use_localhost:
            if framework == InferenceFramework.VLLM:
                response = send_request(f"http://localhost:{local_port}/stream", request)
                response["num_completion_tokens"] = response["payload"]["count_output_tokens"]
            else:
                raise NotImplementedError()
        else:
            response = send_request(
                f"{GATEWAY_URL}/v1/llm/completions-stream?model_endpoint_name={model}",
                request,
                AUTH_USER_ID,
            )
            response["num_completion_tokens"] = response["payload"]["output"][
                "num_completion_tokens"
            ]

        result_queue.put(response)


def generate_request(
    framework: InferenceFramework, prompt: str, output_token_count: int, localhost: bool
):
    if not localhost:
        return {"prompt": prompt, "max_new_tokens": output_token_count, "temperature": 0.0}

    if framework == InferenceFramework.TEXT_GENERATION_INFERENCE:
        return {
            "parameters": {
                "do_sample": False,
                "max_new_tokens": output_token_count,
                "details": False,
            },
            "inputs": prompt,
        }
    elif framework == InferenceFramework.VLLM:
        return {
            "prompt": prompt,
            "max_tokens": output_token_count,
            "temperature": 0,
            "stream": True,
        }
    elif framework == InferenceFramework.LIGHTLLM:
        return {
            "parameters": {
                "do_sample": False,
                "max_new_tokens": output_token_count,
            },
            "inputs": prompt,
        }
    elif framework == InferenceFramework.TENSORRT_LLM:
        return {
            "max_tokens": output_token_count,
            "text_input": prompt,
            "bad_words": "",
            "stop_words": "",
        }
    else:
        raise NotImplementedError()


def send_requests(
    model: str,
    prompt: str,
    output_token_counts: List[int],
    use_localhost: bool,
    concurrency: int,
    framework: InferenceFramework,
    local_port: int = 5005,
):
    thread_results: queue.Queue = queue.Queue()
    requests_queue: queue.Queue = queue.Queue()
    for output_token_count in output_token_counts:
        request = generate_request(framework, prompt, output_token_count, use_localhost)
        requests_queue.put(request)
    threads = []
    for i in range(concurrency):
        thread = threading.Thread(
            target=pull_and_send_request_from_queue,
            args=(
                model,
                requests_queue,
                thread_results,
                use_localhost,
                framework,
                local_port,
            ),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    results = []
    while not thread_results.empty():
        results.append(thread_results.get())

    return results


def generate_prompt(num, hf_model):
    random.seed(1)
    text = lorem.words(num // 2)  # Roughly 2 tokens per lorem word
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    return tokenizer.decode(tokenizer.encode(text)[: num - 2])


def generate_output_token_counts(mean, std, num, input_token_count):
    output = np.random.normal(mean, std, num).astype(int).tolist()

    for i in range(len(output)):
        output[i] = min(output[i], MAX_CONTEXT_WINDOW - input_token_count)
    return output


def run_benchmark(
    model: str,
    framework: InferenceFramework,
    hf_model: str,
    config: BenchmarkConfig,
    num_trials: int,
    use_localhost: bool,
    concurrency: int,
    verbose: bool,
    local_port: int,
):
    group_statistics = []
    prompt = generate_prompt(config.input_token_count, hf_model)

    prompt_num_tokens = config.input_token_count

    output_token_counts = generate_output_token_counts(
        config.output_token_count_mean,
        config.output_token_count_std,
        num_trials,
        config.input_token_count,
    )

    start = time.time()
    results = send_requests(
        model,
        prompt,
        output_token_counts,
        use_localhost,
        concurrency,
        framework,
        local_port=local_port,
    )
    end = time.time()
    elapsed = end - start
    results = [result for result in results if result is not None]

    num_sampled_tokens = sum([result["num_completion_tokens"] for result in results])
    num_prompt_tokens = prompt_num_tokens * len(results)
    time_to_process_prompt = []
    time_per_completion = []
    for result in results:
        avg_time_per_token = (result["total_time"] - result["time_to_first_token"]) / (
            result["num_completion_tokens"] - 1
        )
        time_to_process_prompt.append(result["time_to_first_token"] - avg_time_per_token)
        time_per_completion.append(result["total_time"] - time_to_process_prompt[-1])

    total_num_tokens = num_sampled_tokens + num_prompt_tokens

    n = len(results)

    statistics = {
        "num_prompt_tokens": prompt_num_tokens,
        "avg_num_sampled_tokens": num_sampled_tokens / n,
        "elapsed_time": elapsed,
        "num_requests": num_trials,
        "num_successful_requests": n,
        "total_num_tokens": total_num_tokens,
        "total_num_sampled_tokens": num_sampled_tokens,
        "prompt_throughput": num_prompt_tokens / (sum(time_to_process_prompt) / concurrency),
        "sampling_throughput": num_sampled_tokens / (sum(time_per_completion) / concurrency),
        "per_session_sampling_throughput": num_sampled_tokens / sum(time_per_completion),
    }
    if verbose:
        print(f"Statistics: {statistics}")
    group_statistics.append(statistics)

    # Sleep for 1 seconds between each fixture.
    time.sleep(1)

    return group_statistics


@app.command()
def run_benchmarks(
    model: str,
    framework: str,
    input_token_count: int,
    output_token_count_mean: int,
    num_trials: int = 50,
    output_file: Optional[str] = None,
    use_localhost: bool = False,
    concurrency: int = 1,
    verbose: bool = False,
    hf_model: Optional[str] = None,
    local_port: int = 5005,
):
    """Run benchmarks."""
    all_statistics = []
    config = BenchmarkConfig(input_token_count, output_token_count_mean)
    try:
        if verbose:
            print(f"Running benchmark for config {config}")
        if hf_model is None:
            if model not in HF_MODEL_MAPPING:
                raise ValueError(
                    f"--hf-model must be specified for model {model} since it's not in default mapping."
                )
            hf_model = HF_MODEL_MAPPING[model]
        statistics = run_benchmark(
            model,
            InferenceFramework.from_value(framework),
            hf_model,
            config,
            num_trials,
            use_localhost,
            concurrency,
            verbose,
            local_port,
        )
        all_statistics.extend(statistics)
    except Exception:
        traceback.print_exc()

    if output_file is not None:
        with open(output_file, "w") as f:
            yaml.safe_dump(all_statistics, f)


if __name__ == "__main__":
    typer.run(run_benchmarks)
