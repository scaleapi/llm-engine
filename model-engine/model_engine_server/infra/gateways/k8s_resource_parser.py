import hashlib
import re
from typing import Union

# found this regex floating around somewhere, probably validates k8s requests in general:
# '^([+-]?[0-9.]+)([eEinumkKMGTP]*[-+]?[0-9]*)$'


def validate_cpu_request(req: str):
    return re.match(r"^([0-9]+)m?$|^([0-9]+\.[0-9]+)$", req) is not None


def parse_cpu_request(req: str) -> int:
    """Returns the nearest integer number of millicpus corresponding to the request"""
    if not validate_cpu_request(req):
        raise ValueError(f"{req} isn't a valid k8s cpu request")
    is_millis = "m" in req
    raw_req = float(req.replace("m", ""))
    return round(raw_req * (1 if is_millis else 1e3))


memoryMultipliers = {
    None: 1,
    "k": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
    "P": 1000**5,
    "E": 1000**6,
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "Pi": 1024**5,
    "Ei": 1024**6,
}

memoryRegex = r"^(?P<number>([0-9]+)(\.([0-9]+))?)(?P<suffix>k|M|G|T|P|E|Ki|Mi|Gi|Ti|Pi|Ei)?$"


def validate_mem_request(req: str):
    return re.match(memoryRegex, req) is not None


def parse_mem_request(req: str):
    """Returns the nearest integer in bytes corresponding to the request"""
    match = re.match(memoryRegex, req)
    if match is None:
        raise ValueError(f"{req} isn't a valid k8s memory request")
    multiplier = memoryMultipliers[match.group("suffix")]
    number = float(match.group("number"))
    return int(round(number * multiplier))


def get_node_port(service_name: str) -> int:
    """Hashes the service name to a port number in the range [30000, 32767]"""
    return int(hashlib.sha256(service_name.encode()).hexdigest(), 16) % (32768 - 30000) + 30000


def get_target_concurrency_from_per_worker_value(per_worker: int) -> float:
    """Returns the target concurrency given a per-worker value"""
    return per_worker


def get_per_worker_value_from_target_concurrency(concurrency: Union[str, int, float]) -> int:
    """Returns the per-worker value given a target concurrency.

    Inverse of get_target_concurrency_from_per_worker_value
    """
    return int(round(parse_cpu_request(str(concurrency)) / 1000.0))


def format_bytes(num_bytes) -> str:
    """Convert bytes to human-readable format"""
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}Yi"
