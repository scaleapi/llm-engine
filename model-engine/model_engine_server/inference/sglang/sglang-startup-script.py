#!/usr/bin/env python3
import argparse
import os
import socket
import subprocess
import time


def wait_for_dns(dns_name: str, max_retries: int = 20, sleep_seconds: int = 3):
    """
    Attempt to resolve the given dns_name up to max_retries times,
    sleeping sleep_seconds between attempts.
    Raises RuntimeError if resolution fails repeatedly.
    """
    for attempt in range(1, max_retries + 2):
        try:
            # Use AF_UNSPEC to allow both IPv4 and IPv6
            socket.getaddrinfo(dns_name, None, socket.AF_UNSPEC)
            print(f"[DNS-OK] Successfully resolved {dns_name} on attempt {attempt}.")
            return  # If successful, just return
        except socket.gaierror as e:
            print(
                f"[DNS-FAIL] Attempt {attempt} to resolve {dns_name} failed: {e}. "
                f"Retrying in {sleep_seconds} seconds..."
            )
            time.sleep(sleep_seconds)
    # If we exhaust our attempts, raise an error
    raise RuntimeError(f"Could not resolve {dns_name} after {max_retries} attempts.")


def main(
    model: str,
    node_rank: int,
    nnodes: int,
    tp: int,
    worker_port: int,
    leader_port: int,
    s3_path: str,
):
    # 1) Download the DeepSeek model using s5cmd
    model_path = f"/data/model_files/{model}"
    os.makedirs(model_path, exist_ok=True)

    s5cmd_cmd = [
        "s5cmd",
        "--numworkers=512",
        "cp",
        "--concurrency=10",
        "--include",
        "*.model",
        "--include",
        "*.json",
        "--include",
        "*.safetensors",
        "--include",
        "*.py",
        "--include",
        "tokenizer.model.v*",
        "--exclude",
        "optimizer*",
        f"s3://{s3_path}/{model}/*",
        model_path,
    ]
    print("Running s5cmd download command...")
    subprocess.check_call(s5cmd_cmd)
    print("Download complete.")

    # 2) Wait for both the leader and current Pod DNS to resolve
    leader_pod_dns = f"{os.environ.get('LWS_LEADER_ADDRESS')}.svc.cluster.local"
    current_pod_dns = f"{os.environ.get('K8S_OWN_POD_NAME')}.{os.environ.get('K8S_LWS_NAME')}.{os.environ.get('K8S_OWN_NAMESPACE')}.svc.cluster.local"

    print(f"Waiting for DNS resolution of leader pod: {leader_pod_dns}")
    wait_for_dns(leader_pod_dns)

    print(f"Waiting for DNS resolution of current pod: {current_pod_dns}")
    wait_for_dns(current_pod_dns)

    # 3) Now that we know the leader DNS is resolvable, get the leader's IP
    try:
        addr_info = socket.getaddrinfo(leader_pod_dns, None, socket.AF_UNSPEC)
        ip_addresses = [addr[4][0] for addr in addr_info]
        print(f"Resolved IP addresses for {leader_pod_dns}: {ip_addresses}")
    except socket.gaierror as e:
        print(f"Error resolving {leader_pod_dns}: {e}")
        raise

    # Expect exactly one unique IP from the resolution
    if len(set(ip_addresses)) != 1:
        raise RuntimeError(
            f"Expected a single unique IP address for {leader_pod_dns}, got: {ip_addresses}"
        )
    local_ip = ip_addresses[0]

    # 4) Save the local IP to the env var "SGLANG_HOST_IP"
    os.environ["SGLANG_HOST_IP"] = local_ip
    print(f"SGLANG_HOST_IP environment variable set to {local_ip}")

    # 5) Start the SGLang server
    # See https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L302 for all available args
    print("Starting SGLang server...")
    sglang_cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--tp",
        str(tp),
        "--host",
        "0.0.0.0",
        "--port",
        str(worker_port),
        "--dist-init-addr",
        f"[{local_ip}]:{str(leader_port)}",
        "--nnodes",
        str(nnodes),
        "--node-rank",
        str(node_rank),
        "--trust-remote-code",
        "--log-level",
        "debug",
    ]
    print("Running SGLang server command...")
    subprocess.check_call(sglang_cmd)
    print("SGLang server started.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--worker-port", type=int, default=5005)
    parser.add_argument("--leader-port", type=int, default=5002)
    parser.add_argument("--s3-path", type=str, default="scale-ml/models/hf-synced-weights")
    args = parser.parse_args()
    main(
        model=args.model,
        node_rank=args.node_rank,
        nnodes=args.nnodes,
        tp=args.tp,
        worker_port=args.worker_port,
        leader_port=args.leader_port,
        s3_path=args.s3_path,
    )
