"""会话持久化单元测试 — memory + sqlite 双后端 + 工厂。

Day 7 Part 3 验收点：
  1. memory：进程内往返、覆盖、返回的是副本（不串改内部状态）
  2. sqlite：文件落盘 → 新实例读回（模拟"重启恢复"）
  3. 工厂按 backend 创建 / 未知 backend 抛错
"""

from app.storage.session_store import MemorySessionStore, SqliteSessionStore, create_session_store


def _messages() -> list[dict]:
    return [
        {"role": "user", "content": "怎么退货？"},
        {"role": "assistant", "content": "7 天内可退货。"},
    ]


class TestMemorySessionStore:
    async def test_empty_then_save_then_load(self) -> None:
        store = MemorySessionStore()
        assert await store.load("s1") == []
        await store.save("s1", messages=_messages(), tokens_used=10, finish_reason="stop")
        assert await store.load("s1") == _messages()

    async def test_load_returns_copy_not_internal_state(self) -> None:
        store = MemorySessionStore()
        await store.save("s1", messages=_messages(), tokens_used=1, finish_reason="stop")
        got = await store.load("s1")
        got.append({"role": "user", "content": "篡改"})
        assert await store.load("s1") == _messages()  # 内部状态未被污染

    async def test_save_overwrites_session(self) -> None:
        store = MemorySessionStore()
        await store.save("s1", messages=_messages(), tokens_used=1, finish_reason="stop")
        second = [{"role": "user", "content": "再来一轮"}]
        await store.save("s1", messages=second, tokens_used=2, finish_reason="stop")
        assert await store.load("s1") == second


class TestSqliteSessionStore:
    async def test_persists_across_instances(self, tmp_path) -> None:
        """写进文件，换一个实例读回 —— 等价于进程重启后恢复。"""
        path = str(tmp_path / "sessions.db")
        store1 = SqliteSessionStore(path)
        await store1.save("s1", messages=_messages(), tokens_used=10, finish_reason="stop")

        store2 = SqliteSessionStore(path)  # 模拟重启后的新实例
        assert await store2.load("s1") == _messages()

    async def test_load_missing_session_is_empty(self, tmp_path) -> None:
        store = SqliteSessionStore(str(tmp_path / "s.db"))
        assert await store.load("nope") == []


class TestFactory:
    def test_backend_memory(self) -> None:
        assert isinstance(create_session_store("memory"), MemorySessionStore)

    def test_backend_sqlite(self, tmp_path) -> None:
        assert isinstance(
            create_session_store("sqlite", path=str(tmp_path / "x.db")), SqliteSessionStore
        )

    def test_unknown_backend_raises(self) -> None:
        try:
            create_session_store("redis")  # Day 13 才接，现在应明确报错
        except ValueError:
            return
        raise AssertionError("未知 backend 应抛 ValueError")
