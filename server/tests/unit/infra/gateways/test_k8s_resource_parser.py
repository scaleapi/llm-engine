import pytest

from spellbook_serve.infra.gateways.k8s_resource_parser import (
    get_per_worker_value_from_target_concurrency,
    get_target_concurrency_from_per_worker_value,
    parse_cpu_request,
    parse_mem_request,
    validate_cpu_request,
    validate_mem_request,
)


def test_validate_cpu_request():
    test_cases = [
        ("", False),
        ("42mm", False),
        ("h1", False),
        ("1h1", False),
        ("1n", False),
        ("1..1", False),
        ("4.5.6", False),
        ("1.2m", False),
        (".1", False),
        (".", False),
        ("1", True),
        ("1m", True),
        ("1.5", True),
        ("4.24", True),
        ("123412341241234", True),
    ]
    for item, expected in test_cases:
        assert validate_cpu_request(item) == expected


def test_parse_cpu_request():
    test_cases = [
        ("42", 42000),
        ("1.5", 1500),
        ("300m", 300),
        ("4000m", 4000),
        ("0.1", 100),
    ]
    for item, expected in test_cases:
        assert expected == parse_cpu_request(item)


def test_validate_mem_request():
    test_cases = [
        ("", False),
        ("number", False),
        ("hi42", False),
        ("1j", False),
        ("1m", False),
        ("4.5.6", False),
        ("5Ki123", False),
        ("1", True),
        ("4224", True),
        ("1.99", True),
        ("42424242424", True),
        ("1k", True),
        ("2.1M", True),
        ("3G", True),
        ("4.2T", True),
        ("5P", True),
        ("60.3E", True),
        ("42.5Ki", True),
        ("898Mi", True),
        ("33Gi", True),
        ("44Ti", True),
        ("55Pi", True),
        ("66Ei", True),
    ]
    for item, expected in test_cases:
        assert validate_mem_request(item) == expected


def test_parse_mem_request():
    test_cases = [
        ("1", 1),
        ("4224", 4224),
        ("1.99", 2),
        ("42424242424", 42424242424),
        ("1k", 1 * 1000),
        ("2.1M", int(2.1 * 1000**2)),
        ("3G", 3 * 1000**3),
        ("4.2T", int(4.2 * 1000**4)),
        ("5P", 5 * 1000**5),
        ("60.3E", int(60.3 * 1000**6)),
        ("42.5Ki", int(42.5 * 1024)),
        ("898Mi", 898 * 1024**2),
        ("33Gi", 33 * 1024**3),
        ("44Ti", 44 * 1024**4),
        ("55Pi", 55 * 1024**5),
        ("66Ei", 66 * 1024**6),
    ]
    for item, expected in test_cases:
        assert parse_mem_request(item) == expected


@pytest.mark.parametrize(
    "input_value",
    [
        "1",
        "1.5",
        "500m",
        "5500m",
    ],
)
def test_get_target_concurrency_to_per_worker_value(input_value):
    assert get_target_concurrency_from_per_worker_value(
        parse_cpu_request(str(get_per_worker_value_from_target_concurrency(input_value)))
    ) == parse_cpu_request(input_value)
