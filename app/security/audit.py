"""审计日志 — 谁/何时/对什么/结果，全部留痕（Day 8 Part 4）。

后端照抄 session_store 双后端姿势：
  - memory：list + asyncio.Lock，开发/测试零依赖（默认）
  - sqlite：stdlib sqlite3 文件持久化（同步调用用 asyncio.to_thread 包住）

PII 安全：target/detail 由调用方在传入前脱敏（见 pii.redact）——审计只落"类型/摘要"，
不落裸手机号，否则审计表自己就成了泄露源（AGENTS 3.2）。
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class AuditEntry:
    """一条审计记录。kind 见下；target/detail 传入前须已脱敏。"""

    kind: str  # guard_block | guard_flag | tool_call | tool_denied | pii_redacted
    session_id: str
    actor: str  # user | agent | system
    action: str  # 工具名 / 规则名 / pii 类型
    target: str = ""  # 订单号 / 消息摘要（截断，不落全文）
    detail: str = ""
    created_at: datetime = field(default_factory=_utcnow)


class MemoryAuditLog:
    """内存后端：进程内 list，重启即丢 —— 单测 / 本地零依赖跑。"""

    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []
        self._lock = asyncio.Lock()

    async def record(self, entry: AuditEntry) -> None:
        async with self._lock:
            self.entries.append(entry)
        _log(entry)


class SqliteAuditLog:
    """SQLite 文件后端：可查、可留痕。同步 sqlite3 用 to_thread 包，不阻塞事件循环。"""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_entries (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind       TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    actor      TEXT NOT NULL,
                    action     TEXT NOT NULL,
                    target     TEXT NOT NULL DEFAULT '',
                    detail     TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )

    def _write(self, entry: AuditEntry) -> None:
        with sqlite3.connect(self._path) as con:
            con.execute(
                """
                INSERT INTO audit_entries
                    (kind, session_id, actor, action, target, detail, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.kind,
                    entry.session_id,
                    entry.actor,
                    entry.action,
                    entry.target,
                    entry.detail,
                    entry.created_at.isoformat(),
                ),
            )

    async def record(self, entry: AuditEntry) -> None:
        async with self._lock:
            await asyncio.to_thread(self._write, entry)
        _log(entry)

    async def query(self, session_id: str | None = None) -> list[AuditEntry]:
        """按会话查审计（None = 全部），便于验收"审计能查到拒绝"。"""
        rows = await asyncio.to_thread(self._fetch, session_id)
        return [
            AuditEntry(
                kind=r[0], session_id=r[1], actor=r[2], action=r[3],
                target=r[4], detail=r[5], created_at=datetime.fromisoformat(r[6]),
            )
            for r in rows
        ]

    def _fetch(self, session_id: str | None) -> list[tuple[str, ...]]:
        with sqlite3.connect(self._path) as con:
            if session_id is None:
                cur = con.execute("SELECT * FROM audit_entries ORDER BY id")
            else:
                cur = con.execute(
                    "SELECT * FROM audit_entries WHERE session_id = ? ORDER BY id", (session_id,)
                )
            # SELECT * 含 id 列，去掉它只取后面的字段
            return [tuple(str(c) for c in row[1:]) for row in cur.fetchall()]


def _log(entry: AuditEntry) -> None:
    """审计同时进 structlog（结构化，可被采集/告警）。"""
    data = asdict(entry)
    data.pop("created_at", None)
    logger.info("audit", **data)


def create_audit_log(
    backend: str = settings.audit_backend,
    path: str = settings.audit_path,
) -> MemoryAuditLog | SqliteAuditLog:
    """按配置创建审计后端。新增后端在此扩展。"""
    if backend == "memory":
        return MemoryAuditLog()
    if backend == "sqlite":
        return SqliteAuditLog(path)
    raise ValueError(f"不支持的 audit_backend: {backend}")
