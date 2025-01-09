import csv
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
from lorem_text import lorem
from transformers import AutoTokenizer

AUTH_USER_ID = os.getenv("AUTH_USER_ID")
GATEWAY_URL = os.getenv("GATEWAY_URL")
app = typer.Typer(name="throughput-benchmarks", add_completion=False)

MAX_CONTEXT_WINDOW = 100000


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
    SGLANG = "sglang"

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
    inter_token_latencies = []
    last_token_time = None
    payload_json: dict = {}
    num_completion_tokens = 0  # We calculate this value manually since tensorrt llm doesn't give it
    for byte_payload in response.iter_lines():
        # Skip line
        if byte_payload == b"\n" or byte_payload == b"":
            continue

        token_time = time.time()
        if first_line:
            time_to_first_token = token_time - start
            last_token_time = token_time
            first_line = False
        else:
            inter_token_latencies.append(token_time - last_token_time)
            last_token_time = token_time

        payload = byte_payload.decode("utf-8")

        # Event data
        if payload.startswith("data:"):
            payload_data = payload.lstrip("data:").rstrip("/n")
            if payload_data.lstrip().rstrip() == "[DONE]":
                # Ignore the done message from sglang
                continue
            payload_json = json.loads(payload_data)
        num_completion_tokens += 1

    return {
        "payload": payload_json,
        "time_to_first_token": time_to_first_token,
        "total_time": time.time() - start,
        "inter_token_latencies": inter_token_latencies,
        "num_completion_tokens": num_completion_tokens,
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
                response["num_completion_tokens"] = response["payload"][
                    "count_output_tokens"
                ]  # vLLM gives us completion token count, use that.
            elif framework == InferenceFramework.TENSORRT_LLM:
                response = send_request(
                    f"http://localhost:{local_port}/v2/models/ensemble/generate_stream", request
                )
            elif framework == InferenceFramework.SGLANG:
                response = send_request(f"http://localhost:{local_port}/generate", request)
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
    temperature = 0.0

    if not localhost:
        return {"prompt": prompt, "max_new_tokens": output_token_count, "temperature": temperature}

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
            "temperature": temperature,
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
            "parameters": {
                "temperature": temperature,
                "stream": True,
            },
        }
    elif framework == InferenceFramework.SGLANG:
        return {
            "text": prompt,
            "stream": True,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": output_token_count,
            },
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
    prompts_list_override: Optional[List] = None,
):
    thread_results: queue.Queue = queue.Queue()
    requests_queue: queue.Queue = queue.Queue()
    for i, output_token_count in enumerate(output_token_counts):
        if prompts_list_override is not None:
            new_prompt = prompts_list_override[i % len(prompts_list_override)]
        else:
            new_prompt = prompt
        request = generate_request(framework, new_prompt, output_token_count, use_localhost)
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


def generate_prompt(num, tokenizer):
    text = lorem.words(num // 2)  # Roughly 2 tokens per lorem word
    return tokenizer.decode(tokenizer.encode(text)[: num - 2])


def generate_output_token_counts(mean, std, num, input_token_count):
    output = np.random.normal(mean, std, num).astype(int).tolist()

    for i in range(len(output)):
        output[i] = min(output[i], MAX_CONTEXT_WINDOW - input_token_count)
    return output


def generate_output_token_counts_from_existing(
    distribution: List[int], num: int, input_token_count: int
):
    assert len(distribution) > 0, "Can't have a distribution with 0 tokens"
    output = []
    # Sample without replacement so that we don't have as much variance
    for _ in range(num // len(distribution)):
        random.shuffle(distribution)
        output.extend(distribution)
    random.shuffle(distribution)
    output.extend(distribution[: num % len(distribution)])
    assert len(output) == num

    for i in range(len(output)):
        output[i] = min(output[i], MAX_CONTEXT_WINDOW - input_token_count)
    return output


def read_data_from_json_file(fpath: str):
    # Assumes the distribution is some json-formatted string that represents a list
    try:
        with open(fpath, "r") as fin:
            return json.load(fin)
    except FileNotFoundError:
        print("File not found. Exiting.")
        raise


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
    response_token_count_distribution: Optional[List] = None,
    prompts_list_override: Optional[List] = None,
    generate_distinct_prompts: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    if not generate_distinct_prompts:
        random.seed(1)
    prompt = generate_prompt(config.input_token_count, tokenizer)
    if generate_distinct_prompts:
        assert (
            prompts_list_override is None
        ), "Can't have both distinct generated prompts and override prompts"
        prompts_list_override = [
            generate_prompt(config.input_token_count, tokenizer) for _ in range(num_trials)
        ]

    prompt_num_tokens = config.input_token_count

    if response_token_count_distribution is not None:
        output_token_counts = generate_output_token_counts_from_existing(
            response_token_count_distribution, num_trials, config.input_token_count
        )
    else:
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
        prompts_list_override=prompts_list_override,
    )
    end = time.time()
    elapsed = end - start
    results = [result for result in results if result is not None]

    num_sampled_tokens = sum([result["num_completion_tokens"] for result in results])
    num_prompt_tokens = prompt_num_tokens * len(results)
    n = len(results)
    time_to_process_prompt = []
    time_per_completion = []
    time_to_first_token = []
    inter_token_latency = []  # one value per request, average inter-token latency in the request
    total_request_time = []
    all_inter_token_latencies = []  # one value per token (except the first generated token)
    for result in results:
        avg_time_per_token = (result["total_time"] - result["time_to_first_token"]) / (
            max(1, result["num_completion_tokens"] - 1)
        )
        time_to_first_token.append(result["time_to_first_token"])
        time_to_process_prompt.append(result["time_to_first_token"] - avg_time_per_token)
        time_per_completion.append(result["total_time"] - time_to_process_prompt[-1])
        inter_token_latency.append(avg_time_per_token)
        total_request_time.append(result["total_time"])
        all_inter_token_latencies.extend(result["inter_token_latencies"])

    total_num_tokens = num_sampled_tokens + num_prompt_tokens
    avg_prefill_time = sum(time_to_process_prompt) / n
    avg_completion_time = sum(time_per_completion) / n
    p50_request_time = np.percentile(total_request_time, 50)
    p90_request_time = np.percentile(total_request_time, 90)
    p95_request_time = np.percentile(total_request_time, 95)
    p99_request_time = np.percentile(total_request_time, 99)
    p50_inter_token_latency = np.percentile(all_inter_token_latencies, 50)
    p90_inter_token_latency = np.percentile(all_inter_token_latencies, 90)
    p95_inter_token_latency = np.percentile(all_inter_token_latencies, 95)
    p99_inter_token_latency = np.percentile(all_inter_token_latencies, 99)
    p999_inter_token_latency = np.percentile(all_inter_token_latencies, 99.9)
    p50_time_to_first_token = np.percentile(time_to_first_token, 50)
    p90_time_to_first_token = np.percentile(time_to_first_token, 90)
    p95_time_to_first_token = np.percentile(time_to_first_token, 95)
    p99_time_to_first_token = np.percentile(time_to_first_token, 99)

    statistics = {
        "concurrency": concurrency,
        "avg_prompt_throughput": num_prompt_tokens
        / (elapsed * avg_prefill_time / (avg_prefill_time + avg_completion_time)),
        "avg_time_to_first_token": sum(time_to_first_token) / n,
        "p50_time_to_first_token": p50_time_to_first_token,
        "p90_time_to_first_token": p90_time_to_first_token,
        "p95_time_to_first_token": p95_time_to_first_token,
        "p99_time_to_first_token": p99_time_to_first_token,
        "avg_sampling_throughput": num_sampled_tokens
        / (elapsed * avg_completion_time / (avg_prefill_time + avg_completion_time)),
        "avg_total_throughput": total_num_tokens / elapsed,
        "avg_per_session_sampling_throughput": num_sampled_tokens
        / (elapsed * avg_completion_time / (avg_prefill_time + avg_completion_time))
        / concurrency,
        "avg_request_throughput": n / elapsed,
        "avg_inter_token_latency": sum(inter_token_latency) / n,
        "p50_inter_token_latency": p50_inter_token_latency,
        "p90_inter_token_latency": p90_inter_token_latency,
        "p95_inter_token_latency": p95_inter_token_latency,
        "p99_inter_token_latency": p99_inter_token_latency,
        "p99.9_inter_token_latency": p999_inter_token_latency,
        "num_prompt_tokens": prompt_num_tokens,
        "avg_num_sampled_tokens": num_sampled_tokens / n,
        "elapsed_time": elapsed,
        "avg_prefill_time": avg_prefill_time,
        "avg_completion_time": avg_completion_time,
        "p50_request_time": p50_request_time,
        "p90_request_time": p90_request_time,
        "p95_request_time": p95_request_time,
        "p99_request_time": p99_request_time,
        "num_requests": num_trials,
        "num_successful_requests": n,
        "total_num_tokens": total_num_tokens,
        "total_num_sampled_tokens": num_sampled_tokens,
    }
    if verbose:
        print(f"Statistics: {statistics}")

    # Sleep for 1 seconds between each benchmark.
    time.sleep(1)

    return statistics


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
    response_token_count_distribution_file: Optional[str] = None,
    prompts_list_override_file: Optional[str] = None,
    generate_distinct_prompts: bool = False,
):
    """Run benchmarks."""
    all_statistics = []
    config = BenchmarkConfig(input_token_count, output_token_count_mean)

    response_token_count_distribution = None
    if response_token_count_distribution_file is not None:
        response_token_count_distribution = read_data_from_json_file(
            response_token_count_distribution_file
        )
    prompts_list_override = None
    if prompts_list_override_file is not None:
        prompts_list_override = read_data_from_json_file(prompts_list_override_file)

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
            response_token_count_distribution,
            prompts_list_override,
            generate_distinct_prompts,
        )
        all_statistics.append(statistics)
    except Exception:
        traceback.print_exc()

    if output_file is not None:
        header = all_statistics[0].keys()
        import os

        if not os.path.exists(output_file):
            with open(output_file, "w") as csvfile:
                print("creating the data in csv")
                csv_writer = csv.DictWriter(csvfile, fieldnames=header)
                csv_writer.writeheader()
                csv_writer.writerows(all_statistics)
        else:
            with open(output_file, "a") as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=header)
                csv_writer.writerows(all_statistics)


@app.command()
def run_benchmarks_concurrency_range(
    model: str,
    framework: str,
    input_token_count: int,
    output_token_count_mean: int,
    num_trials_per_concurrency: int = 5,
    output_file: Optional[str] = None,
    use_localhost: bool = False,
    concurrency_min: int = 1,
    concurrency_max: int = 1,
    concurrency_step: int = 1,
    verbose: bool = False,
    hf_model: Optional[str] = None,
    local_port: int = 5005,
    response_token_count_distribution_file: Optional[str] = None,
    prompts_list_override_file: Optional[str] = None,
    generate_distinct_prompts: bool = False,
):
    for concurrency in range(concurrency_min, concurrency_max + 1, concurrency_step):
        run_benchmarks(
            model,
            framework,
            input_token_count,
            output_token_count_mean,
            num_trials_per_concurrency * concurrency,
            output_file,
            use_localhost,
            concurrency,
            verbose,
            hf_model,
            local_port,
            response_token_count_distribution_file,
            prompts_list_override_file,
            generate_distinct_prompts,
        )


if __name__ == "__main__":
    app()
