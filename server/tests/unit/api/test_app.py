from fastapi.testclient import TestClient


def test_healthcheck(simple_client: TestClient):
    response = simple_client.get("/healthcheck")
    assert response.status_code == 200

    response = simple_client.get("/healthz")
    assert response.status_code == 200

    response = simple_client.get("/readyz")
    assert response.status_code == 200
