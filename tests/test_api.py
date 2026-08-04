"""API 集成测试 — 验证服务可启动、健康检查、mock 聊天。"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check() -> None:
    """健康检查返回 ok。"""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_chat_mock_success() -> None:
    """正常聊天请求返回 mock 响应。"""
    resp = client.post("/api/v1/chat", json={"message": "怎么退货？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"]
    assert "[Mock]" in body["answer"]
    assert body["model"] == "mock (not connected)"


def test_chat_empty_message_rejected() -> None:
    """空消息返回 422（参数校验）。"""
    resp = client.post("/api/v1/chat", json={"message": ""})
    assert resp.status_code == 422


def test_chat_stream_not_implemented() -> None:
    """流式接口返回 501（Day 5 实现）。"""
    resp = client.get("/api/v1/chat/session-1/stream")
    assert resp.status_code == 501
