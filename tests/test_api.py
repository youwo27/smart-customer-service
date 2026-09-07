"""API 集成测试 — 服务可启动、健康检查、真管道聊天（注入假 orchestrator）。

Day 7 起 /chat 不再返回 [Mock]，而是走 app.state.orchestrator；测试注入假编排层，
不连 DeepSeek / 不检索，就能验证 HTTP 层 → 编排层的接线。
"""

from typing import Any

from fastapi.testclient import TestClient

from app.core.agent import AgentRunResult
from app.core.events import AgentEvent
from app.main import app

client = TestClient(app)


class _FakeOrchestrator:
    """假编排层：按会话累计轮数返回确定性回答。

    传了 emit 时，像真 Orchestrator 一样先把一条 assistant + end 事件推给
    事件回调（SSE 路由靠它逐帧下发），再返回结果。
    """

    def __init__(self) -> None:
        self.turns: dict[str, int] = {}
        self.calls: list[tuple[str, str]] = []

    async def handle(
        self, message: str, session_id: str, emit: Any = None
    ) -> AgentRunResult:
        self.calls.append((session_id, message))
        n = self.turns.get(session_id, 0) + 1
        self.turns[session_id] = n
        if emit is not None:
            for event in (
                AgentEvent(
                    kind="assistant",
                    session_id=session_id,
                    content=f"（假编排）第 {n} 条回复",
                ),
                AgentEvent(
                    kind="end",
                    session_id=session_id,
                    content="stop",
                    data={"tokens_used": 10},
                ),
            ):
                res = emit(event)
                if res is not None:
                    await res
        return AgentRunResult(
            messages=[{"role": "assistant", "content": f"（假编排）第 {n} 条回复"}],
            tool_calls_count=0,
            tokens_used=10,
            model="fake",
            finish_reason="stop",
        )


_fake = _FakeOrchestrator()
app.state.orchestrator = _fake


def test_health_check() -> None:
    """健康检查返回 ok。"""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_routes_through_orchestrator() -> None:
    """/chat 已接入真管道：不再返回 [Mock]，回复来自 orchestrator。"""
    resp = client.post("/api/v1/chat", json={"message": "怎么退货？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"]
    assert "假编排" in body["answer"] and "[Mock]" not in body["answer"]
    assert body["model"] == "fake"
    assert body["tool_calls_count"] == 0


def test_chat_multi_turn_reuses_session() -> None:
    """同一 session_id 多轮：编排层能看到累计轮数（历史确实被带上）。"""
    sid = "e2e-session-1"
    first = client.post("/api/v1/chat", json={"message": "能退吗", "session_id": sid})
    second = client.post("/api/v1/chat", json={"message": "那运费呢", "session_id": sid})
    assert first.status_code == 200 and second.status_code == 200
    assert "第 1 条" in first.json()["answer"]
    assert "第 2 条" in second.json()["answer"]
    assert first.json()["session_id"] == sid == second.json()["session_id"]
    assert len(_fake.calls) >= 2
    assert _fake.calls[-1] == (sid, "那运费呢")


def test_chat_empty_message_rejected() -> None:
    """空消息返回 422（参数校验）。"""
    resp = client.post("/api/v1/chat", json={"message": ""})
    assert resp.status_code == 422


def test_chat_stream_emits_sse_events() -> None:
    """POST /chat/stream 返回 text/event-stream，事件经假编排逐帧下发到 end。"""
    resp = client.post("/api/v1/chat/stream", json={"message": "怎么退货？"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "event: assistant" in body
    assert "event: end" in body
    # data 是单行 JSON，含 session_id
    assert "session_id" in body and '"kind": "assistant"' in body
