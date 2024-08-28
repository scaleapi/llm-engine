import csv
import json
import queue
import random
import threading
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import typer
from lorem_text import lorem
from transformers import AutoTokenizer

"""
python throughput_benchmark.py --num-trials=100 --output-file=out.txt --raw-file=raw.txt --concurrency=20
"""

AUTH_KEY = API_KEY = None


class AuthMode(Enum):
    X_API_KEY = "x-api-key"
    BASIC = "basic"


# local url
ENDPOINT_URL = f"http://localhost:5005/predict"

VERBOSE = False
app = typer.Typer(name="throughput-benchmarks", add_completion=False)

MAX_CONTEXT_WINDOW = 512
PROMPT_SIZE = 512

SINGLE_REQUEST_TIMEOUT = 90  # They impose a timeout of 90 seconds for sync
SINGLE_REQUEST_STREAMING_TIMEOUT = (
    300  # I'm gonna impose a timeout of 300 seconds for streaming
)


def generate_prompt(num, hf_model):
    random.seed(random.randint(0, 1000000))
    text = lorem.words(num // 2)  # Roughly 2 tokens per lorem word
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    return tokenizer.decode(tokenizer.encode(text)[: num - 2])


def send_request(url, request):
    start = time.time()
    if isinstance(request["text"], list):
        url = url + "-batch"
    response = requests.post(
        url,
        json=request,
        # stream=True,
        headers={"x-api-key": API_KEY} if API_KEY else {},
        auth=(AUTH_KEY, "") if AUTH_KEY else None,
    )
    time_to_first_token = None
    inter_token_latencies = []
    payload_json = None
    num_prompt_tokens = 0
    num_completion_tokens = 0
    # stream = request["stream"]
    stream = False

    time_to_first_token = time.time() - start
    try:
        payload_json = response.json()
    except:
        payload_json = {}
    if VERBOSE:
        print("Sent request", request)
        print("Got response", payload_json.get("output", {}).get("text", ""))
    # Need to see if we get back num_prompt_tokens
    # (Outdated) Unfortunately we don't get back number of prompt tokens through MIG
    # need new version of chihuahua wrapper for prompt tokens
    num_prompt_tokens = 0
    num_completion_tokens = 0
    # num_prompt_tokens = payload_json.get("output", {}).get("num_prompt_tokens", 0)
    # num_completion_tokens = payload_json.get("output", {}).get(
    #     "num_completion_tokens", 0
    # )

    status_code = response.status_code
    # handle 500
    raw_response = ""
    if status_code != 200:
        if not stream:
            print("Error response", response.text)
            raw_response = response.text
        else:
            print("Error while streaming", payload_json)

    print("status code", status_code)
    end = time.time()
    # TODO prompt tokens maybe

    return {
        "payload": payload_json,
        "raw_response": raw_response,  # Recording this just in case we want to go and look at non-200 responses
        "total_time": end - start,
        "time_to_first_token": time_to_first_token,
        "inter_token_latencies": inter_token_latencies,
        "status_code": status_code,
        "start_time": start,
        "end_time": end,
        "num_prompt_tokens": num_prompt_tokens,
        "num_completion_tokens": num_completion_tokens,
    }


def pull_and_send_request_from_queue(
    request_queue: queue.Queue,
    result_queue: queue.Queue,
):
    while not request_queue.empty():
        request = request_queue.get()
        response = send_request(
            ENDPOINT_URL,
            request,
        )

        result_queue.put(response)


def send_requests(
    requests: List[Any],
    concurrency: int,
):
    thread_results: queue.Queue = queue.Queue()
    requests_queue: queue.Queue = queue.Queue()
    for r in requests:
        requests_queue.put(r)

    threads = []
    for _ in range(concurrency):
        thread = threading.Thread(
            target=pull_and_send_request_from_queue,
            args=(
                requests_queue,
                thread_results,
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


def run_benchmark(
    num_trials: int,
    concurrency: int,
    verbose: bool,
    batch_size: int = 1,
    hf_model: str = "ProtectAI/deberta-v3-base-prompt-injection",
):
    random.seed(1)
    prompts = [generate_prompt(PROMPT_SIZE, hf_model)] * num_trials
    # group prompts into batches
    if batch_size == 1:
        requests = [{"text": prompt} for prompt in prompts]
    else:
        requests = [
            {
                "text": (
                    prompts[i : i + batch_size]
                    if i + batch_size < num_trials
                    else prompts[i:]
                )
            }
            for i in range(0, num_trials, batch_size)
        ]

    start = time.time()
    results = send_requests(
        requests,
        concurrency,
    )
    end = time.time()
    elapsed = end - start
    results = [result for result in results if result is not None]

    completion_tokens = [
        result["num_completion_tokens"]
        for result in results
        if "num_completion_tokens" in result
        and result["num_completion_tokens"] is not None
    ]
    num_sampled_tokens = sum(completion_tokens)
    n = len(results)
    # time_to_process_prompt = []
    # time_per_completion = []
    time_to_first_token = []
    inter_token_latency = (
        []
    )  # one value per request, average inter-token latency in the request
    total_request_time = []
    status_codes = []
    all_inter_token_latencies = (
        []
    )  # one value per token (except the first generated token)
    for result in results:
        #     avg_time_per_token = (result["total_time"] - result["time_to_first_token"]) / (
        #         result["num_completion_tokens"] - 1
        #     )
        avg_time_per_token = 0
        if result.get("inter_token_latencies", []):
            avg_time_per_token = sum(result["inter_token_latencies"]) / len(
                result["inter_token_latencies"]
            )
        time_to_first_token.append(result["time_to_first_token"])
        #     time_to_process_prompt.append(result["time_to_first_token"] - avg_time_per_token)
        #     time_per_completion.append(result["total_time"] - time_to_process_prompt[-1])
        inter_token_latency.append(avg_time_per_token)
        total_request_time.append(result["total_time"])
        all_inter_token_latencies.extend(result.get("inter_token_latencies", []))
        status_codes.append(result["status_code"])
    if len(all_inter_token_latencies) == 0:
        all_inter_token_latencies = [0]  # Dummy value to avoid div/0

    # total_num_tokens = num_sampled_tokens + num_prompt_tokens
    # avg_prefill_time = sum(time_to_process_prompt) / n
    # avg_completion_time = sum(time_per_completion) / n
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

    p50_completion_tokens = np.percentile(completion_tokens, 50)
    p90_completion_tokens = np.percentile(completion_tokens, 90)
    p95_completion_tokens = np.percentile(completion_tokens, 95)
    p99_completion_tokens = np.percentile(completion_tokens, 99)
    completion_tokens_lt_256 = len([ct for ct in completion_tokens if ct < 256])
    completion_tokens_lt_512 = len([ct for ct in completion_tokens if ct < 512])
    completion_tokens_lt_1024 = len([ct for ct in completion_tokens if ct < 1024])
    completion_tokens_ge_1024 = len([ct for ct in completion_tokens if ct >= 1024])

    num_successful_requests = len(
        [status_code for status_code in status_codes if status_code == 200]
    )

    statistics = {
        "concurrency": concurrency,
        # "avg_prompt_throughput": num_prompt_tokens
        # / (elapsed * avg_prefill_time / (avg_prefill_time + avg_completion_time)),
        "avg_time_to_first_token": sum(time_to_first_token) / n,
        "p50_time_to_first_token": p50_time_to_first_token,
        "p90_time_to_first_token": p90_time_to_first_token,
        "p95_time_to_first_token": p95_time_to_first_token,
        "p99_time_to_first_token": p99_time_to_first_token,
        # "avg_sampling_throughput": num_sampled_tokens
        # / (elapsed * avg_completion_time / (avg_prefill_time + avg_completion_time)),
        # # "avg_total_throughput": total_num_tokens / elapsed,
        # "avg_per_session_sampling_throughput": num_sampled_tokens
        # / (elapsed * avg_completion_time / (avg_prefill_time + avg_completion_time))
        # / concurrency,
        "avg_request_throughput": n / elapsed,
        "avg_inter_token_latency": sum(inter_token_latency) / n,
        "p50_inter_token_latency": p50_inter_token_latency,
        "p90_inter_token_latency": p90_inter_token_latency,
        "p95_inter_token_latency": p95_inter_token_latency,
        "p99_inter_token_latency": p99_inter_token_latency,
        "p99.9_inter_token_latency": p999_inter_token_latency,
        # "num_prompt_tokens": prompt_num_tokens,
        "avg_num_sampled_tokens": num_sampled_tokens / n,
        "elapsed_time": elapsed,
        # "avg_prefill_time": avg_prefill_time,
        # "avg_completion_time": avg_completion_time,
        "p50_request_time": p50_request_time,
        "p90_request_time": p90_request_time,
        "p95_request_time": p95_request_time,
        "p99_request_time": p99_request_time,
        "p50_completion_tokens": p50_completion_tokens,
        "p90_completion_tokens": p90_completion_tokens,
        "p95_completion_tokens": p95_completion_tokens,
        "p99_completion_tokens": p99_completion_tokens,
        "completion_tokens_lt_256": completion_tokens_lt_256,
        "completion_tokens_lt_512": completion_tokens_lt_512,
        "completion_tokens_lt_1024": completion_tokens_lt_1024,
        "completion_tokens_ge_1024": completion_tokens_ge_1024,
        "num_requests": num_trials,
        "num_successful_requests": num_successful_requests,
        # "total_num_tokens": total_num_tokens,
        "total_num_sampled_tokens": num_sampled_tokens,
    }
    if verbose:
        print(f"Statistics: {statistics}")

    # Sleep for 1 seconds between each benchmark.
    time.sleep(1)

    return statistics, results


@app.command()
def run_benchmarks(
    num_trials: int = 50,
    output_file: Optional[str] = None,
    raw_file: Optional[str] = None,
    batch_size: int = 1,
    concurrency: int = 1,
    verbose: bool = False,
    extra_verbose: bool = False,
    seed: Optional[int] = None,
):
    """Run benchmarks."""
    all_statistics = []

    if seed is not None:
        # Set constant seed for reproducibility.
        np.random.seed(seed)
        random.seed(seed)

    global VERBOSE

    try:
        if verbose:
            print("Running benchmark")
        if extra_verbose:
            VERBOSE = True  # Hacky way of not needing to pass verbose around
        statistics, results = run_benchmark(
            num_trials,
            concurrency,
            verbose,
            batch_size=batch_size,
        )
        all_statistics.append(statistics)
    except Exception:
        traceback.print_exc()

    if output_file is not None:
        header = all_statistics[0].keys()

        with open(output_file, "a") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=header)
            csv_writer.writeheader()
            csv_writer.writerows(all_statistics)

    if raw_file is not None:
        df = pd.DataFrame.from_records(results)
        df.to_csv(raw_file, index=False)


# @app.command()
def run_benchmarks_bulk(
    output_file: Optional[str] = None,
    raw_file: Optional[str] = None,
    verbose: bool = False,
    extra_verbose: bool = False,
    seed: Optional[int] = None,
):
    """Run benchmarks."""
    all_statistics = []
    # (num_trials, concurrency)
    settings = [
        (1000, 10),
        (1000, 20),
        (1000, 30),
        (1000, 40),
        (1000, 50),
        (1000, 60),
    ]

    if seed is not None:
        # Set constant seed for reproducibility.
        np.random.seed(seed)
        random.seed(seed)

    global VERBOSE

    try:
        for setting in settings:
            if verbose:
                print(
                    f"Running benchmark with {setting[0]} trials and {setting[1]} concurrency"
                )
            if extra_verbose:
                VERBOSE = True  # Hacky way of not needing to pass verbose around

            statistics, results = run_benchmark(
                num_trials=setting[0], concurrency=setting[1], verbose=verbose
            )
            all_statistics.append(statistics)
    except Exception:
        traceback.print_exc()

    if output_file is not None:
        header = all_statistics[0].keys()

        with open(output_file, "a") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=header)
            csv_writer.writeheader()
            csv_writer.writerows(all_statistics)

    if raw_file is not None:
        df = pd.DataFrame.from_records(results)
        df.to_csv(raw_file, index=False)


if __name__ == "__main__":
    app()
