# TODO this should initialize multinode and ray. Should look similar to the set up for multinode online serving.
# TODO a few things differ here, we need to look at a bunch of env vars to determine if 1. we're leader, 2. ray address, port, cluster_size, own_address.
# In one case, we have JOB_COMPLETION_INDEX, NUM_INSTANCES, MASTER_ADDR, MASTER_PORT as available env vars.
# (May) need to get own_address from somewhere. In serving, it's from a few env vars, and is a k8s dns name.

import argparse
import os
import subprocess
import sys
import time

RAY_INIT_TIMEOUT = 1200


def start_worker(ray_address, ray_port, ray_init_timeout):
    for i in range(0, ray_init_timeout, 5):
        result = subprocess.run(
            [
                "ray",
                "start",
                "--address",
                f"{ray_address}:{ray_port}",
                "--block",
                # "--node-ip-address",
                # own_address,
            ],
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"Worker: Ray runtime started with head address {ray_address}:{ray_port}")
            sys.exit(0)
        print(result.returncode)
        print("Waiting until the ray worker is active...")
        time.sleep(5)
    print(f"Ray worker starts timeout, head address: {ray_address}:{ray_port}")
    sys.exit(1)


def start_leader(ray_port, ray_cluster_size, ray_init_timeout):
    subprocess.run(
        ["ray", "start", "--head", "--port", str(ray_port)]  # , "--node-ip-address", own_address]
    )
    for i in range(0, ray_init_timeout, 5):
        active_nodes = subprocess.run(
            [
                "python3",
                "-c",
                'import ray; ray.init(); print(sum(node["Alive"] for node in ray.nodes()))',
            ],
            capture_output=True,
            text=True,
        )
        active_nodes = int(active_nodes.stdout.strip())
        if active_nodes == ray_cluster_size:
            print("All ray workers are active and the ray cluster is initialized successfully.")
            sys.exit(0)
        print(f"Wait for all ray workers to be active. {active_nodes}/{ray_cluster_size} is active")
        time.sleep(5)
    print("Waiting for all ray workers to be active timed out.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Ray cluster initialization script")
    parser.add_argument("--ray_init_timeout", type=int, default=RAY_INIT_TIMEOUT)

    args = parser.parse_args()

    is_leader = os.getenv("JOB_COMPLETION_INDEX") == "0"
    ray_address = os.getenv("MASTER_ADDR")
    ray_port = os.getenv("MASTER_PORT")
    ray_cluster_size = os.getenv("NUM_INSTANCES")

    if is_leader:
        start_leader(ray_port, ray_cluster_size, args.ray_init_timeout)
    else:
        start_worker(ray_address, ray_port, args.ray_init_timeout)


if __name__ == "__main__":
    main()
