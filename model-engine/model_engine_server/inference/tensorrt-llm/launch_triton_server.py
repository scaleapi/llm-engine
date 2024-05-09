import argparse
import subprocess
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=1, help="world size, only support tensor parallelism now"
    )
    parser.add_argument("--tritonserver", type=str, default="/opt/tritonserver/bin/tritonserver")
    parser.add_argument(
        "--http-address",
        type=str,
        default="ipv6:[::1]",
        help="Default HTTP address to ipv6:[::1].",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=5005,
        help="Default HTTP port to 5005. See llm-engine/model-engine/model_engine_server/inference/configs/service--http_forwarder.yaml",
    )
    path = str(Path(__file__).parent.absolute()) + "/../all_models/gpt"
    parser.add_argument("--model_repo", type=str, default=path)
    return parser.parse_args()


def get_cmd(world_size, tritonserver, model_repo, http_address, http_port):
    cmd = "mpirun --allow-run-as-root "
    for i in range(world_size):
        cmd += f" -n 1 {tritonserver} --model-repository={model_repo} --http-address {http_address} --http-port {http_port} --disable-auto-complete-config --backend-config=python,shm-region-prefix-name=prefix{i}_ : "
    return cmd


if __name__ == "__main__":
    args = parse_arguments()
    cmd = get_cmd(
        int(args.world_size), args.tritonserver, args.model_repo, args.http_address, args.http_port
    )
    subprocess.call(cmd, shell=True)
