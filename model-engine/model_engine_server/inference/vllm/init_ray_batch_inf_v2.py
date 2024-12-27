import socket
import subprocess
import time

import ray

RETRY_INTERVAL_SEC = 5


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
    Wait until the cluster reaches the expected size.

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

            print(f"Current cluster size: {current_size}/{expected_nodes} nodes")

            if current_size >= expected_nodes:
                print("Cluster reached expected size!")
                return True

            # Print status of nodes that aren't alive
            if len(nodes) != len(alive_nodes):
                dead_nodes = [node for node in nodes if not node["Alive"]]
                for node in dead_nodes:
                    print(
                        f"Node {node['NodeID']} is not alive: {node.get('LastError', 'Unknown error')}"
                    )

        except Exception as e:
            print(f"Error checking cluster size: {e}")

        time.sleep(check_interval)

    print(f"Timeout waiting for cluster to reach size {expected_nodes}")
    return False


def start_leader(
    ray_port: int,
    node_ip_address: str,
) -> bool:
    # node ip address in this case is actually a DNS name for the pod
    result = subprocess.run(
        ["ray", "start", "--head", "--port", str(ray_port), "--node-ip-address", node_ip_address]
    )
    if result.returncode == 0:
        print(f"Leader: Ray runtime started with port {ray_port}")
        return True
    print(f"Failed to start Ray leader node with port {ray_port}")
    return False


def start_worker(
    ray_port: int,
    node_ip_address: str,
    leader_addr: str,
    timeout: int,
) -> bool:
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
            print(f"Worker: Ray runtime started with head address {leader_addr}:{ray_port}")
            return True
        print(result.returncode)
        print("Waiting until the ray worker is active...")
        time.sleep(5)
    print(f"Ray worker starts timeout, head address: {leader_addr}:{ray_port}")
    return False


def init_ray(
    leader_addr: str,
    leader_port: int,
    is_leader: bool,
    cluster_size: int,
    # node_ip_address: Optional[str] = None,
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
    # TODO figure out if this thing works for node_ip_address
    # if node_ip_address is None:
    node_ip_address = (
        socket.gethostname() + "." + leader_addr.split(".", 1)[1]
    )  # dumb hack to get the equivalent of leader_addr

    print(f"Waiting for head node DNS ({leader_addr}) to be resolvable...")
    head_ip_info = wait_for_dns(leader_addr, timeout=timeout)
    if head_ip_info is None:
        raise RuntimeError(f"Timeout waiting for DNS resolution of {leader_addr}")

    # ray_params = {
    #     "_node_ip_address": node_ip_address,
    # }

    # if not is_leader:
    #     ray_params["address"] = (
    #         f"ray://{leader_addr}:{leader_port}"
    #     )

    # ray.init(**ray_params)  # TODO replace with a subprocess call since we can't set port via ray.init
    if is_leader:
        if not start_leader(leader_port, node_ip_address):
            raise RuntimeError("Failed to start Ray leader node")
    else:
        if not start_worker(leader_port, node_ip_address, leader_addr, timeout):
            raise RuntimeError("Failed to start Ray worker node")
    print(
        f"Successfully initialized Ray {'head' if is_leader else 'worker'} node at {node_ip_address}"
    )

    # After successful initialization, wait for cluster to reach expected size
    if is_leader and not wait_for_cluster_nodes(cluster_size, timeout=timeout):
        raise RuntimeError(
            f"Cluster did not reach expected size of {cluster_size} nodes within {timeout} seconds"
        )

    return
