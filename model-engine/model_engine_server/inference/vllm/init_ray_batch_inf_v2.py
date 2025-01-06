import argparse
import socket
import subprocess
import sys
import time

import ray

RETRY_INTERVAL_SEC = 5


def get_node_ip_address(leader_addr: str) -> str:
    # Assumes we're on a K8s cluster where the leader address is
    # <leader pod name>.<the rest of the FQDN>
    # e.g. if we're using a JobSet
    # Kinda of a dumb hack to get an externally addressable DNS name
    node_ip_address = socket.gethostname() + "." + leader_addr.split(".", 1)[1]
    return node_ip_address


def wait_for_dns(hostname: str, timeout: int = 300, interval: int = 5):
    """
    Wait for DNS resolution of the hostname.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            ip = socket.getaddrinfo(hostname, None)
            return ip
        except socket.gaierror:
            print(f"Waiting for DNS resolution of {hostname}...")
            time.sleep(interval)
    return None


def wait_for_cluster_nodes(
    expected_nodes: int, timeout: int = 600, check_interval: int = 10
) -> bool:
    """
    Wait until the cluster reaches the expected size. Runs in head node

    Args:
        expected_nodes: Expected number of nodes in the cluster
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds

    Returns:
        bool: True if cluster reached expected size, False if timeout occurred
    """
    # Since we've subprocess.run for starting ray, need to connect in the cluster right here.
    ray.init()
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            nodes = ray.nodes()
            alive_nodes = [node for node in nodes if node["Alive"]]
            current_size = len(alive_nodes)

            print(f"Current cluster size: {current_size}/{expected_nodes} nodes", flush=True)

            if current_size >= expected_nodes:
                print("Cluster reached expected size!", flush=True)
                return True

            # Print status of nodes that aren't alive
            if len(nodes) != len(alive_nodes):
                dead_nodes = [node for node in nodes if not node["Alive"]]
                for node in dead_nodes:
                    print(
                        f"Node {node['NodeID']} is not alive: {node.get('LastError', 'Unknown error')}",
                        flush=True,
                    )

        except Exception as e:
            print(f"Error checking cluster size: {e}", flush=True)

        time.sleep(check_interval)

    print(f"Timeout waiting for cluster to reach size {expected_nodes}", flush=True)
    return False


# The functions wait_for_head_node_to_exit and wait_for_head_node_to_exit_process
# are used to ensure that the worker node exits successfully when the head node is no longer reachable
# wait_for_head_node_to_exit_process empirically returns a non-zero status code,
# which is caught by the wait_for_head_node_to_exit function and eventually exits
# with a zero status code.
# We do this so that upstream job infrastructure can read a zero status code as a successful job completion.

# The order of execution is as follows:
# 1. wait_for_head_node_to_exit() is called from some script in the worker node
# 2. wait_for_head_node_to_exit() spawns a subprocess that runs this file.
# 3. This file calls wait_for_head_node_to_exit_process()
# 4. wait_for_head_node_to_exit_process() runs in the subprocess and waits until the head node is no longer reachable, then exits with a non-zero status code
# 5. wait_for_head_node_to_exit() catches the non-zero status code and returns
# 6. The upstream script returns successfully


async def wait_for_head_node_to_exit() -> None:
    # Runs in worker node
    # Waits until there is no longer a connection to the head Ray node, then exits
    # This name needs to equal the name of the file!
    # Also, if this file gets moved to a different directory in the docker image, the path needs to be updated
    worker_process = subprocess.Popen(
        [sys.executable, "init_ray_batch_inf_v2.py", "--mode", "wait_for_head_node_to_exit"]
    )
    return_code = worker_process.wait()
    if return_code != 0:
        print("Worker terminated with code:", return_code)


def wait_for_head_node_to_exit_process():
    # Runs in worker node
    # This will run in the subprocess spawned and will conveniently error out
    # when the head node is no longer reachable
    # The exit gets caught by the `wait_for_head_node_to_exit` function
    ray.init()
    while True:
        nodes = ray.nodes()
        print(f"Able to get nodes list {len(nodes)}", flush=True)
        time.sleep(RETRY_INTERVAL_SEC)


def start_leader(
    ray_port: int,
    node_ip_address: str,
) -> bool:
    # Runs in head node
    # node ip address in this case is actually a DNS name for the pod
    result = subprocess.run(
        ["ray", "start", "--head", "--port", str(ray_port), "--node-ip-address", node_ip_address]
    )
    if result.returncode == 0:
        print(f"Leader: Ray runtime started with port {ray_port}", flush=True)
        return True
    print(f"Failed to start Ray leader node with port {ray_port}", flush=True)
    return False


def start_worker(
    ray_port: int,
    node_ip_address: str,
    leader_addr: str,
    timeout: int,
) -> bool:
    # Runs in worker
    # node ip address in this case is actually a DNS name for the pod
    start_time = time.time()
    while time.time() - start_time < timeout:
        result = subprocess.run(
            [
                "ray",
                "start",
                "--address",
                f"{leader_addr}:{ray_port}",
                "--node-ip-address",
                node_ip_address,
            ],
            capture_output=True,
        )
        if result.returncode == 0:
            print(
                f"Worker: Ray runtime started with head address {leader_addr}:{ray_port}",
                flush=True,
            )
            return True
        print(
            f"Failed to start Ray worker node with head address {leader_addr}:{ray_port}",
            flush=True,
        )
        print(result.returncode)
        print("Waiting until the ray worker is active...", flush=True)
        time.sleep(5)
    print(f"Ray worker starts timeout, head address: {leader_addr}:{ray_port}", flush=True)
    return False


def init_ray(
    leader_addr: str,
    leader_port: int,
    is_leader: bool,
    cluster_size: int,
    timeout: int = 600,
) -> None:
    """
    Initialize a Ray cluster and wait for all nodes to join.

    Args:
        leader_addr: DNS name of the master node (K8s service)
        leader_port: Port number of the master node
        is_leader: If this is the leader node.
        cluster_size: Expected total number of nodes in the cluster
        node_ip_address: IP address of the current node. If None, will be automatically detected
        timeout: Maximum time to wait for cluster to reach expected size
    """
    node_ip_address = get_node_ip_address(leader_addr)

    print(f"Waiting for head node DNS ({leader_addr}) to be resolvable...", flush=True)
    head_ip_info = wait_for_dns(leader_addr, timeout=timeout)
    if head_ip_info is None:
        raise RuntimeError(f"Timeout waiting for DNS resolution of {leader_addr}")

    if is_leader:
        if not start_leader(leader_port, node_ip_address):
            raise RuntimeError("Failed to start Ray leader node")
    else:
        if not start_worker(leader_port, node_ip_address, leader_addr, timeout):
            raise RuntimeError("Failed to start Ray worker node")
    print(
        f"Successfully initialized Ray {'head' if is_leader else 'worker'} node at {node_ip_address}",
        flush=True,
    )

    # After successful initialization, wait for cluster to reach expected size
    if is_leader and not wait_for_cluster_nodes(cluster_size, timeout=timeout):
        raise RuntimeError(
            f"Cluster did not reach expected size of {cluster_size} nodes within {timeout} seconds"
        )

    return


# The following allows the worker to spawn a subprocess to wait for the head node to exit


def main(mode: str):
    if mode == "wait_for_head_node_to_exit":
        wait_for_head_node_to_exit_process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["wait_for_head_node_to_exit"], required=True)
    args = parser.parse_args()
    main(args.mode)
