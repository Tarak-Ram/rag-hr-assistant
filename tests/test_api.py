from fastapi.testclient import TestClient

from app.main import app


def test_health_and_chat():
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["indexed_chunks"] > 0

        res = client.post("/chat", json={"question": "What is the daily allowance for domestic travel?", "debug": True})
        assert res.status_code == 200
        body = res.json()
        assert "2,500" in body["answer"]
        assert body["citations"] and body["trace"]


def test_rejects_empty_question():
    with TestClient(app) as client:
        assert client.post("/chat", json={"question": ""}).status_code == 422
