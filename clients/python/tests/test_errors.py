import pytest
from llmengine.errors import UnknownError, parse_error


@pytest.mark.parametrize("content", [b"", b"[]", b"\xff\xfe", b'{"detail": "moved"}'])
def test_redirect_error_explains_base_path_for_any_body(content):
    error = parse_error(302, content)
    assert isinstance(error, UnknownError)
    assert "LLM_ENGINE_BASE_PATH" in str(error)
