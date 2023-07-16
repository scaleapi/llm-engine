"""Utility functions for Python programs. Should not be used by other modules in this package."""
import os
import platform
from functools import lru_cache
from tempfile import NamedTemporaryFile
from typing import Callable, Iterable, Optional, Sequence, Tuple, TypeVar

from llm_engine_server.core.aws.storage_client import sync_storage_client
from llm_engine_server.core.config import ml_infra_config
from llm_engine_server.core.utils.url import parse_attachment_url

In = TypeVar("In")
"""Type variable representing the function under test's input type.
"""

Out = TypeVar("Out")
"""Type variable representing the function under test's output type.
"""

__all__: Sequence[str] = (
    "table_tester",
    "no_aws_r_creds",
    "no_aws_rw_creds",
    "env_var_is_true",
)


def table_tester(
    fn: Callable[[In], Out],
    i_o_pairs: Iterable[Tuple[In, Out]],
    equality: Callable[[Out, Out], bool] = lambda a, b: a == b,
) -> None:
    """Convenience function to apply a function against a series of input & expected output pairs.
    This function `assert`s that the function applied to each input results in the associated output
    value, where equality is checked by the :param:`equality` function, which defaults to Python's `==`.
    """
    for i, (inp, expected) in enumerate(i_o_pairs):
        msg_part = f"Failed on test pair # {i + 1}:\nINPUT:    {inp}\nEXPECTED: {expected}\n"
        try:
            actual = fn(inp)
        except Exception:  # pylint: disable=broad-except
            print(msg_part)
            raise
        assert equality(actual, expected), msg_part + f"ACTUAL:   {actual}"


@lru_cache(1)
def no_aws_r_creds() -> bool:
    """True if we don't have the read AWS access credentials to run tests. False means we do.

    Useful in a `@pytest.mark.skipif(condition=no_aws_r_creds(), reason="No AWS read credentials")`
    marker on a `test_` unittest function.
    """
    return _no_aws_creds(write_check=False)


@lru_cache(1)
def no_aws_rw_creds() -> bool:
    """True if we don't have the read+write AWS access credentials to run tests. False means we do.

    Useful in a `@pytest.mark.skipif(condition=no_aws_rw_creds(), reason="No AWS read/write credentials")`
    marker on a `test_` unittest function.
    """
    return _no_aws_creds(write_check=True)


def _no_aws_creds(*, write_check: bool) -> bool:
    try:
        p = parse_attachment_url(f"s3://{ml_infra_config().s3_bucket}/testing/_keep_do_not_delete")
        s3_client = sync_storage_client()
        if not _exists(s3_client, p):
            return True

        with NamedTemporaryFile() as f:
            f.close()
            # test read
            with open(f.name, "wb") as wb:
                s3_client.download_fileobj(
                    Bucket=p.bucket,
                    Key=p.key,
                    Fileobj=wb,
                )
            if write_check:
                # test write
                with open(f.name, "rb") as rb:
                    s3_client.upload_fileobj(
                        Fileobj=rb,
                        Bucket=p.bucket,
                        Key=p.key,
                    )
    except Exception:  # pylint: disable=broad-except
        return True
    else:
        return False


def _exists(s3_client, p):
    try:
        # https://stackoverflow.com/questions/33842944/check-if-a-key-exists-in-a-bucket-in-s3-using-boto3
        s3_client.head_object(Bucket=p.bucket, Key=p.key)
    except Exception as e:  # type: ignore
        try:
            # pylint: disable=no-member
            error_code = e.response["Error"]["Code"].strip()  # type: ignore
            if error_code in ("404", "NoSuchKey"):
                return False
        except (NameError, KeyError):
            pass
        raise e
    else:
        return True


def env_var_is_true(env_var_name: str) -> bool:
    """Return true if the environment variable is currently set to a known truth value.

    True if the :param:`env_var_name` environment variable is present and contains a truth value.
    The **only** accepted truth values are, case-insensitive:
     - 'y'
     - 'yes'
     - 'true'
     -
    All other values are considered false.
    Additionally, an unset environment variable will result in this function evaluating to false.
    """
    if len(env_var_name) == 0:
        raise ValueError("Need non-empty environment variable name!")

    try:
        x: Optional[str] = os.environ.get(env_var_name, None)
        if x is None:
            return False
        x = x.lower().strip()
        return x in ("y", "true", "yes")
    except Exception:  # pylint: disable=broad-except
        return False


def is_linux() -> bool:
    return "Linux" in platform.platform()
