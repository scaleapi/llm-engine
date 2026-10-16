import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    "files",
    [
        pytest.param({"file": ("test.txt", b"content")}, id="with_file"),
        pytest.param(None, id="without_file"),
    ],
)
def test_upload_file_returns_410(files, test_api_key: str, simple_client: TestClient):
    response = simple_client.post("/v1/files", files=files, auth=(test_api_key, ""))
    assert response.status_code == 410
    assert response.json() == {"detail": "This API is deprecated and no longer available."}
