"""ToolRegistry + 客服工具 的单元测试。"""

import pytest

from app.tools.builtin.customer_service import (
    ApplyRefundTool,
    QueryLogisticsTool,
    QueryOrderTool,
)
from app.tools.registry import ToolRegistry


class TestToolRegistry:
    """ToolRegistry 注册 / 获取 / 列出 / 执行。"""

    def test_register_and_get(self) -> None:
        reg = ToolRegistry()
        tool = QueryOrderTool()
        reg.register(tool)
        assert reg.get("query_order") is tool

    def test_register_duplicate_raises(self) -> None:
        reg = ToolRegistry()
        reg.register(QueryOrderTool())
        with pytest.raises(ValueError, match="重复注册"):
            reg.register(QueryOrderTool())

    def test_get_unknown_raises(self) -> None:
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.get("not_exist")

    def test_list_api_format(self) -> None:
        """list() 输出必须能被 chat_with_tools 认（含 type/function/name/parameters）。"""
        reg = ToolRegistry()
        reg.register(QueryOrderTool())
        tools = reg.list()
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "query_order"
        assert tools[0]["function"]["description"]
        assert tools[0]["function"]["parameters"]["required"] == ["order_id"]

    async def test_execute_unknown_tool_returns_error(self) -> None:
        """未知工具 execute → 结构化错误，不抛异常。"""
        reg = ToolRegistry()
        result = await reg.execute("not_exist")
        assert "error" in result


class TestQueryOrderTool:
    """query_order 工具。"""

    async def test_found(self) -> None:
        result = await QueryOrderTool().execute(order_id="12345")
        assert result["status"] == "已发货"

    async def test_not_found(self) -> None:
        result = await QueryOrderTool().execute(order_id="99999")
        assert "error" in result


class TestQueryLogisticsTool:
    """query_logistics 工具。"""

    async def test_found(self) -> None:
        result = await QueryLogisticsTool().execute(tracking_no="SF1234567890")
        assert "配送站" in result["trace"]

    async def test_not_found(self) -> None:
        result = await QueryLogisticsTool().execute(tracking_no="NOPE")
        assert "error" in result


class TestApplyRefundTool:
    """apply_refund 工具（危险操作）。"""

    def test_declares_permission(self) -> None:
        """危险操作必须声明 required_permissions（Day 8 校验钩子）。"""
        assert "order.refund" in ApplyRefundTool.required_permissions

    async def test_missing_params(self) -> None:
        result = await ApplyRefundTool().execute(order_id="12345", amount=100)
        assert "error" in result

    async def test_shipped_order_goes_to_review(self) -> None:
        """已发货订单退款 → 待审核，不直接成功。"""
        result = await ApplyRefundTool().execute(order_id="12345", amount=899, reason="不想要了")
        assert "审核" in result["status"]

    async def test_unshipped_order_refund_ok(self) -> None:
        """未发货订单退款 → 直接成功。"""
        result = await ApplyRefundTool().execute(order_id="88888", amount=1299, reason="拍错了")
        assert result["status"] == "退款成功"
