# Ignore lint errors for f-strings because the f-strings are actually regex expressions.
# flake8: noqa: W605
import importlib.util
import os
import re
from pathlib import Path
from textwrap import dedent

import pytest
from _pytest.assertion.rewrite import AssertionRewritingHook

from .rest_api_utils import BASE_PATH, SERVICE_IDENTIFIER

ROOT_DIR = Path(__file__).parent.parent

TEST_SKIP_MAGIC_STRING = "# test='skip'"


@pytest.fixture
def tmp_work_path(tmp_path: Path):
    """
    Create a temporary working directory.
    """
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)

    yield tmp_path

    os.chdir(previous_cwd)


class SetEnv:
    def __init__(self):
        self.envars = set()

    def __call__(self, name, value):
        self.envars.add(name)
        os.environ[name] = value

    def clear(self):
        for n in self.envars:
            os.environ.pop(n)


@pytest.fixture
def env():
    setenv = SetEnv()

    yield setenv

    setenv.clear()


@pytest.fixture()
def integration_test_user_id() -> str:
    return os.getenv("TEST_USER_ID", "fakeuser")


def modify_source(source: str) -> str:
    """Adds some custom logic to update code from docs to comply with some requirements."""

    # Ensure the correct base path is used
    source = re.sub(
        r"get_launch_client\((.*)\)\n",
        rf'get_launch_client(\g<1>, gateway_endpoint="{BASE_PATH}")\n',
        source,
    )
    source = re.sub(
        r"LaunchClient\((.*)\)\n",
        rf'LaunchClient(\g<1>, endpoint="{BASE_PATH}")\n',
        source,
    )

    # Add suffix to avoid name collisions
    source = re.sub(
        r"('endpoint_name'|\"endpoint_name\"): ('([\w-]+)'|\"([\w-]+)\")",
        rf"'endpoint_name': '\g<3>\g<4>-{SERVICE_IDENTIFIER}'",
        source,
    )
    source = re.sub(
        r"endpoint_name=('([\w-]+)'|\"([\w-]+)\")",
        rf"endpoint_name='\g<2>\g<3>-{SERVICE_IDENTIFIER}'",
        source,
    )
    source = re.sub(
        r"get_model_endpoint\(\"([\w-]+)\"\)",
        rf'get_model_endpoint("\g<1>-{SERVICE_IDENTIFIER}")',
        source,
    )

    # Set particular tag values
    source = re.sub(r"('team'|\"team\"): ('\w+'|\"\w+\")", r"'team': 'infra'", source)
    source = re.sub(
        r"('product'|\"product\"): ('\w+'|\"\w+\")",
        r"'product': 'launch-integration-test'",
        source,
    )

    source = re.sub(r'"repository": "..."', '"repository": "launch_rearch"', source)
    source = re.sub(
        r'"tag": "..."', '"tag": "11d9d42047cc9a0c6435b19e5e91bc7e0ad31efc-cpu"', source
    )
    source = re.sub(
        r'"command": ...',
        """"command": [
            "dumb-init",
            "--",
            "ddtrace-run",
            "run-service",
            "--config",
            "/install/launch_rearch/config/service--user_defined_code.yaml",
            "--concurrency",
            "1",
            "--http",
            "production",
            "--port",
            "5005",
        ]""",
        source,
    )
    source = re.sub(
        r'"streaming_command": ...',
        """"streaming_command": [
            "dumb-init",
            "--",
            "ddtrace-run",
            "run-streamer",
            "--config",
            "/install/std-ml-srv/tests/resources/example_echo_streaming_service_configuration.yaml",
            "--concurrency",
            "1",
            "--http-mode",
            "production",
            "--port",
            "5005",
        ]""",
        source,
    )
    return source


@pytest.fixture
def import_execute(request, tmp_work_path: Path):
    def _import_execute(module_name: str, source: str, rewrite_assertions: bool = False):
        if rewrite_assertions:
            loader = AssertionRewritingHook(config=request.config)
            loader.mark_rewrite(module_name)
        else:
            loader = None

        module_path = tmp_work_path / f"{module_name}.py"
        modified_source = modify_source(source)
        module_path.write_text(modified_source)
        spec = importlib.util.spec_from_file_location("__main__", str(module_path), loader=loader)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")

    return _import_execute


def extract_code_chunks(path: Path, text: str, offset: int):
    rel_path = path.relative_to(ROOT_DIR)
    for m_code in re.finditer(r"```(.*?)$\n(.*?)\n( *)```", text, flags=re.M | re.S):
        prefix = m_code.group(1).lower()
        if not prefix.startswith(("py", "{.py")):
            continue

        start_line = offset + text[: m_code.start()].count("\n") + 1
        code = dedent(m_code.group(2))
        end_line = start_line + code.count("\n") + 1
        source = "\n" * start_line + code
        if TEST_SKIP_MAGIC_STRING in prefix or TEST_SKIP_MAGIC_STRING in code:
            source = "__skip__"
        yield pytest.param(
            f"{path.stem}_{start_line}_{end_line}", source, id=f"{rel_path}:{start_line}-{end_line}"
        )


def generate_code_chunks(*directories: str):
    for d in directories:
        for path in (ROOT_DIR / d).glob("**/*"):
            if path.suffix == ".py":
                code = path.read_text()
                for m_docstring in re.finditer(r'(^\s*)r?"""$(.*?)\1"""', code, flags=re.M | re.S):
                    start_line = code[: m_docstring.start()].count("\n")
                    docstring = m_docstring.group(2)
                    yield from extract_code_chunks(path, docstring, start_line)
            elif path.suffix == ".md":
                # TODO: remove this hack to skip llms.md
                if "llms.md" in path.name:
                    continue
                code = path.read_text()
                yield from extract_code_chunks(path, code, 0)


# Assumes that launch-python-client is cloned at `models/launch-python-client`
@pytest.mark.parametrize(
    "module_name,source_code",
    generate_code_chunks(
        "launch-python-client/docs",
        "launch-python-client/launch",
        "launch_internal/docs",
        "launch_internal/launch_internal",
    ),
)
def test_docs_examples(
    module_name,
    source_code,
    import_execute,
    env,
    integration_test_user_id,
):
    if source_code == "__skip__":
        pytest.skip("test='skip' on code snippet")

    env("LAUNCH_API_KEY", os.getenv("LAUNCH_TEST_API_KEY", integration_test_user_id))

    try:
        import_execute(module_name, source_code, True)
    except Exception:
        raise
