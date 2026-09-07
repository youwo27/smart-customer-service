"""工具：query_order / query_logistics / apply_refund — 客服业务三件套（mock DB）。

Day 4 用内存 mock 数据演示；真实项目里这里换成查订单 DB / 物流 API / 退款服务。
"""

from typing import Any

from app.tools.base import BaseToolImpl

# ============ mock 数据（真实项目换成 DB / 外部 API） ============

MOCK_ORDERS: dict[str, dict[str, Any]] = {
    "12345": {"status": "已发货", "logistics": "SF1234567890", "amount": 899.0},
    "88888": {"status": "待发货", "logistics": "", "amount": 1299.0},
    "66666": {"status": "已完成", "logistics": "YT778899", "amount": 459.0},
}

MOCK_LOGISTICS: dict[str, str] = {
    "SF1234567890": "【上海市】已到达韵达配送站，今日送达",
    "YT778899": "【深圳市】已签收，感谢使用",
}


class QueryOrderTool(BaseToolImpl):
    """查询订单状态。当用户询问订单到哪了、发货没有、订单状态时使用。参数 order_id 是订单编号。"""

    name = "query_order"
    description = (
        "查询订单状态。当用户询问订单到哪了、发货没有、订单状态时使用。"
        "参数 order_id 是订单编号（纯数字）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "订单编号"},
        },
        "required": ["order_id"],
    }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        order_id = kwargs.get("order_id", "")
        info = MOCK_ORDERS.get(order_id)
        if info is None:
            return {"order_id": order_id, "error": "未找到该订单，请核对订单号"}
        return {"order_id": order_id, "status": info["status"]}


class QueryLogisticsTool(BaseToolImpl):
    """查询物流轨迹。当用户询问快递走到哪了、物流进度时使用。参数 tracking_no 是快递单号。"""

    name = "query_logistics"
    description = (
        "查询物流轨迹。当用户询问快递走到哪了、物流进度时使用。"
        "参数 tracking_no 是快递单号（通常一串字母+数字）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "tracking_no": {"type": "string", "description": "快递单号"},
        },
        "required": ["tracking_no"],
    }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        tracking_no = kwargs.get("tracking_no", "")
        trace = MOCK_LOGISTICS.get(tracking_no)
        if trace is None:
            return {"tracking_no": tracking_no, "error": "未找到该单号的物流信息"}
        return {"tracking_no": tracking_no, "trace": trace}


class ApplyRefundTool(BaseToolImpl):
    """提交退款申请。危险操作：必须用户明确要求才调用，声明 required_permissions 留权限钩子。"""

    name = "apply_refund"
    description = (
        "为用户提交退款申请。**仅当用户明确要求退款**且提供了订单号、退款金额、退款原因时才调用。"
        "不要主动建议退款。参数 amount 是退款金额（元）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "订单编号"},
            "amount": {"type": "number", "description": "退款金额（元）"},
            "reason": {"type": "string", "description": "退款原因"},
        },
        "required": ["order_id", "amount", "reason"],
    }
    required_permissions = ["order.refund"]  # 危险操作：Day 8 权限校验的钩子

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        order_id = kwargs.get("order_id", "")
        amount = kwargs.get("amount")
        reason = kwargs.get("reason", "")

        if not order_id or amount is None or not reason:
            return {"error": "退款申请参数不完整：需要 order_id、amount、reason"}

        if order_id not in MOCK_ORDERS:
            return {"order_id": order_id, "error": "未找到该订单，无法退款"}

        # mock：已发货订单需走售后审核（真实项目这里对接退款服务）
        if MOCK_ORDERS[order_id]["status"] in ("已发货", "已完成"):
            return {
                "order_id": order_id,
                "amount": amount,
                "status": "已提交，待售后审核",
                "message": "订单已发货，退款申请已提交售后审核，1-3 个工作日内处理",
            }
        return {
            "order_id": order_id,
            "amount": amount,
            "status": "退款成功",
            "message": "退款将原路退回，预计 1-7 个工作日到账",
        }
