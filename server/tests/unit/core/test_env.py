import os
from typing import Any, Callable, Dict, Optional, Sequence
from uuid import uuid4

import pytest

from llm_engine_server.core.utils.env import environment

# DO NOT EXPORT ANYTHING
__all__: Sequence[str] = ()


def expect_not_present(e: str) -> None:
    assert (
        e not in os.environ
    ), f"Not expecting env var {e} to be present, instead found {os.environ[e]}"


def expect_present(e: str, value: Any) -> None:
    assert e in os.environ, f"Expecting env var {e} to be present with {value}"
    assert (
        os.environ[e] == value
    ), f"Expected env var {e} to have value {value} but instead found {os.environ[e]}"


def prepare(e: str, existing: Optional[str]) -> Callable[[], None]:
    if existing is not None:
        os.environ[e] = existing
        return lambda: expect_present(e, existing)
    else:
        return lambda: expect_not_present(e)


def test_environment_kwarg():
    e = "ENV_VAR_TEST"
    expect_not_present(e)
    # NOTE: This is to test keyword argument use.
    #       Make sure this **literal value** is the same as `e`'s contents.
    with environment(ENV_VAR_TEST="x"):
        expect_present(e, "x")
    expect_not_present(e)


@pytest.mark.parametrize("existing", ["env var has prior value", None])
def test_environment_normal_cases(existing):
    e = f"___{uuid4()}-test_env_var"
    check = prepare(e, existing)

    check()
    new = f"{uuid4()}--hello_world"
    with environment(**{e: new}):
        expect_present(e, new)
    check()


@pytest.mark.parametrize("existing", ["env var has prior value", None])
def test_environment_with_exception(existing):
    e = f"___{uuid4()}-test_env_var"
    check = prepare(e, existing)

    check()
    new = f"{uuid4()}--hello_world"
    with pytest.raises(ValueError):
        with environment(**{e: new}):
            expect_present(e, new)
            raise ValueError("Uh oh! Something went wrong in our context!")
    check()


def test_environment_multi():
    env_vars: Dict[str, str] = {f"___{uuid4()}-test_env_var--{i}": f"value_{i}" for i in range(25)}

    def ok():
        for e in env_vars.keys():
            expect_not_present(e)

    ok()
    with environment(**env_vars):
        for e, v in env_vars.items():
            expect_present(e, v)
    ok()


def test_environment_invalid_states():
    with pytest.raises(ValueError):
        environment(**{"": "2"})


def test_environment_unset():
    k = f"___{uuid4()}___--test_unset_env_var--"
    v = "hello world! :)"
    # when there is a previous value
    try:
        os.environ[k] = v
        with environment(**{k: None}):
            assert k not in os.environ
        assert k in os.environ
        assert os.environ[k] == v
    finally:
        del os.environ[k]

    # when there is not a previous value
    assert k not in os.environ
    with environment(**{k: None}):
        assert k not in os.environ
    assert k not in os.environ
