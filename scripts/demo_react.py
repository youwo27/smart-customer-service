"""最小 ReAct Demo — 让 DeepSeek 自己决定调用工具。

运行方式:
    python scripts/demo_react.py "查一下订单 12345 的状态"

这里演示了 Agent 的最核心机制：
    模型看到问题 → 决定调用哪个工具 → 拿到结果 → 组织最终回答

Day 4 会把这段逻辑正式实现进 app/core/agent.py。
"""

import asyncio
import json
import os
import sys

# 让 scripts/ 里的脚本能 import 到项目根目录的 app 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Windows 终端 GBK 编码不支持部分 Unicode（emoji 等），改用 UTF-8 输出，避免崩溃
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings
from app.llm.client import LLMClientFactory, LLMConfigFactory

# ============ 1. 定义两个"客服工具" ============
# 真实项目中这些会写在 app/tools/builtin/，这里用 dict 演示

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "查询订单状态。当用户询问订单到哪了、发货没有、订单状态时使用。参数 order_id 是订单编号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单编号"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_logistics",
            "description": "查询物流轨迹。当用户询问快递走到哪了、物流进度时使用。参数 tracking_no 是快递单号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_no": {"type": "string", "description": "快递单号"},
                },
                "required": ["tracking_no"],
            },
        },
    },
]

# ============ 2. 工具的实际执行逻辑（模拟数据） ============

MOCK_ORDERS = {
    "12345": {"status": "已发货", "logistics": "SF1234567890"},
    "88888": {"status": "待发货"},
}

MOCK_LOGISTICS = {
    "SF1234567890": "【上海市】已到达韵达配送站，今日送达",
}


async def execute_tool(name: str, args: dict) -> str:
    """执行工具并返回结果字符串。"""
    if name == "query_order":
        oid = args.get("order_id", "")
        info = MOCK_ORDERS.get(oid)
        return json.dumps({"order_id": oid, "status": info["status"] if info else "未找到该订单"}, ensure_ascii=False)
    if name == "query_logistics":
        tno = args.get("tracking_no", "")
        info = MOCK_LOGISTICS.get(tno)
        return json.dumps({"tracking_no": tno, "trace": info if info else "未找到物流信息"}, ensure_ascii=False)
    return json.dumps({"error": f"未知工具 {name}"})


# ============ 3. 最小 ReAct 循环 ============

SYSTEM_PROMPT = "你是家居电商客服助手，请用简短友好的中文回答用户问题。"


async def run_demo(user_message: str) -> None:
    client = LLMClientFactory.create(LLMConfigFactory.from_settings())

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    print(f"\n===== 用户：{user_message} =====\n")

    # ReAct 循环：最多 5 轮
    for turn in range(5):
        resp = await client.chat_with_tools(messages, TOOLS)
        messages.append({"role": "assistant", "content": resp.content, "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
            for tc in resp.tool_calls
        ]} if resp.tool_calls else {"role": "assistant", "content": resp.content})

        # 模型想调工具 → 执行并回填结果，继续循环
        if resp.tool_calls:
            for tc in resp.tool_calls:
                result = await execute_tool(tc.name, tc.arguments)
                print(f"[tool] 调用 {tc.name}({tc.arguments}) → {result}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            continue

        # 模型没想调工具 → 这是最终回答
        print(f"[answer] {resp.content}")
        print(f"[usage]  输入 {resp.usage.get('input_tokens',0)} tokens / 输出 {resp.usage.get('output_tokens',0)} tokens")
        print(f"[model]  {resp.model}")
        break


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "查一下订单 12345 的状态"
    asyncio.run(run_demo(msg))
