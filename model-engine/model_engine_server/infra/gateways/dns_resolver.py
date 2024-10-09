import socket
from typing import Union


def resolve_dns(host: str, port: Union[str, int] = "http") -> str:
    """
    Returns an IP address of the given host, e.g. "256.256.256.256" for IPv4, or
    "[0000:0000:0000::0000]" for IPv6. You should be able to just substitute this into a URL.
    """
    addrinfo = socket.getaddrinfo(host, port)
    if len(addrinfo) == 0:
        raise ValueError("Host not found.")
    # Probably just need the first one
    socket_type = addrinfo[0][0]
    ip = addrinfo[0][4][0]
    # Do I want to do anything with port? it probably ends up being the default (e.g. 80 for http, 443 for https)
    if socket_type == socket.AF_INET6:
        return f"[{ip}]"
    elif socket_type == socket.AF_INET:
        return ip
    else:
        raise ValueError("Unknown socket type.")
