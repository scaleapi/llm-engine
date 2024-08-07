# TODO this should initialize multinode and ray. Should look similar to the set up for multinode online serving.
# TODO a few things differ here, we need to look at a bunch of env vars to determine if 1. we're leader, 2. ray address, port, cluster_size, own_address.


import argparse
import subprocess
import sys
import time


def start_worker(ray_address, ray_port, ray_init_timeout, own_address):
    for i in range(0, ray_init_timeout, 5):
        result = subprocess.run(
            [
                "ray",
                "start",
                "--address",
                f"{ray_address}:{ray_port}",
                "--block",
                "--node-ip-address",
                own_address,
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


def start_leader(ray_port, ray_cluster_size, ray_init_timeout, own_address):
    subprocess.run(
        ["ray", "start", "--head", "--port", str(ray_port), "--node-ip-address", own_address]
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
    subparsers = parser.add_subparsers(dest="subcommand")

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--ray_address", required=True)
    worker_parser.add_argument("--ray_port", type=int, default=6379)
    worker_parser.add_argument("--ray_init_timeout", type=int, default=1200)
    worker_parser.add_argument("--own_address", required=True)

    leader_parser = subparsers.add_parser("leader")
    leader_parser.add_argument("--ray_port", type=int, default=6379)
    leader_parser.add_argument("--ray_cluster_size", type=int, required=True)
    leader_parser.add_argument("--ray_init_timeout", type=int, default=1200)
    leader_parser.add_argument("--own_address", required=True)

    args = parser.parse_args()

    if args.subcommand == "worker":
        start_worker(args.ray_address, args.ray_port, args.ray_init_timeout, args.own_address)
    elif args.subcommand == "leader":
        start_leader(args.ray_port, args.ray_cluster_size, args.ray_init_timeout, args.own_address)
    else:
        print("unknown subcommand:", args.subcommand)
        sys.exit(1)


if __name__ == "__main__":
    main()
