"""会话持久化 — 自研 checkpointer（Day 7 实现）。

存什么（回答 Day 6 Part 5 理解检查"AgentLoop 断了要恢复要序列化哪些东西"）：
  - 完整 messages（不含 system —— system 每次由 assembler 重建）
  - tokens_used / finish_reason 等元信息
后端：
  - memory：dict + asyncio.Lock，开发/测试零依赖（默认）
  - sqlite：stdlib sqlite3 文件持久化 = 重启恢复；同步调用用 asyncio.to_thread 包住
Day 13 换 Postgres/Redis 时实现同一接口即可，编排层不用改。
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from app.config import settings


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class SessionSnapshot:
    """一次会话的持久化快照（load 返回 messages；save 整份覆盖）。"""

    session_id: str
    messages: list[dict[str, Any]]
    tokens_used: int = 0
    finish_reason: str = ""
    summary: str = ""
    updated_at: datetime = field(default_factory=_utcnow)


class MemorySessionStore:
    """内存后端：进程内 dict，重启即丢 —— 单测 / 本地零依赖跑。"""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionSnapshot] = {}
        self._lock = asyncio.Lock()

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        snap = self._sessions.get(session_id)
        return [] if snap is None else [dict(m) for m in snap.messages]

    async def save(
        self,
        session_id: str,
        *,
        messages: list[dict[str, Any]],
        tokens_used: int,
        finish_reason: str,
    ) -> None:
        async with self._lock:
            self._sessions[session_id] = SessionSnapshot(
                session_id=session_id,
                messages=[dict(m) for m in messages],
                tokens_used=tokens_used,
                finish_reason=finish_reason,
            )


class SqliteSessionStore:
    """SQLite 文件后端：真正的"重启恢复"。同步 sqlite3 用 to_thread 包，不阻塞事件循环。"""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._init_db()  # 启动时一次性建表（毫秒级，不阻塞事件循环）

    def _init_db(self) -> None:
        with sqlite3.connect(self._path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id    TEXT PRIMARY KEY,
                    messages      TEXT NOT NULL,
                    tokens_used   INTEGER NOT NULL DEFAULT 0,
                    finish_reason TEXT NOT NULL DEFAULT '',
                    updated_at    TEXT NOT NULL
                )
                """
            )

    def _fetch(self, session_id: str) -> str | None:
        with sqlite3.connect(self._path) as con:
            row = con.execute("SELECT messages FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return None if row is None else row[0]

    def _write(self, session_id: str, messages: str, tokens_used: int, finish_reason: str) -> None:
        with sqlite3.connect(self._path) as con:
            con.execute(
                """
                INSERT INTO sessions (session_id, messages, tokens_used, finish_reason, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    messages=?, tokens_used=?, finish_reason=?, updated_at=?
                """,
                (
                    session_id, messages, tokens_used, finish_reason, _utcnow().isoformat(),
                    messages, tokens_used, finish_reason, _utcnow().isoformat(),
                ),
            )

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        raw = await asyncio.to_thread(self._fetch, session_id)
        if raw is None:
            return []
        return cast(list[dict[str, Any]], json.loads(raw))

    async def save(
        self,
        session_id: str,
        *,
        messages: list[dict[str, Any]],
        tokens_used: int,
        finish_reason: str,
    ) -> None:
        payload = json.dumps(messages, ensure_ascii=False)
        async with self._lock:
            await asyncio.to_thread(self._write, session_id, payload, tokens_used, finish_reason)


def create_session_store(
    backend: str = settings.session_store_backend,
    path: str = settings.session_store_path,
) -> MemorySessionStore | SqliteSessionStore:
    """按配置创建会话存储后端。新增后端（Day 13 的 Postgres/Redis）在此扩展。"""
    if backend == "memory":
        return MemorySessionStore()
    if backend == "sqlite":
        return SqliteSessionStore(path)
    raise ValueError(f"不支持的 session_store_backend: {backend}")
