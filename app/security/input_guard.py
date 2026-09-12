r"""输入守卫 — L1 规则层（确定性、零 LLM、可单测）。

分层理由（纵深防御，见 guardrails.py 的 InputGuard 组合）：
  - L1 规则：确定性、零成本、可离线单测。命中即硬拦截（block），谁都不放行。
    —— 这是"最后一道永不失效的网"，不依赖模型聪明不聪明。
  - L2 分类：语义级，能抓规则表没写的变体；但贵、可能错，只做"标记"（flag），
    不轻易 block——宁可让疑似注入进去被动作权限兜住，也不因分类器误杀正常请求。

fail 语义：L1 命中 → block（fail-closed，可用性受损无所谓、安全优先）。
正则是"特征码"，攻击者改写措辞就躲过 → 所以它只是第一层，下面还有 L2。

中文无空格，故用 `.{0,N}` 表达"若干字符内"而非 `\s*`；N 取到 12-16 保证命中，
同时每条规则都带**具体攻击特征词**，避免误伤正常客服文本（"帮我查订单12345" 不能命中）。
"""

import re
from dataclasses import dataclass

# 已知攻击模式：正则是"特征码"，被攻击者改写就躲过 → 所以它只是第一层。
# 每条规则都带**具体攻击特征词**，避免误伤正常客服文本（"帮我查订单12345" 不能命中）。
_REGEX_FLAGS = re.IGNORECASE | re.MULTILINE

REGEX_RULES: list[tuple[str, str]] = [
    # 1. 套取/无视系统人设与指令
    ("system_prompt_leak", r"忽略.{0,12}(指令|提示词|提示|人设|设定)"),
    # 2. 角色劫持：让 Agent 不再是客服 / 变成其他角色
    ("role_override", r"(从现在起|现在开始|从现在开始|你一直都是).{0,6}(你是|你就是|扮演|成为)|(不再|已经不是).{0,4}(是|扮演)?(客服|助手|客服机器人)"),
    # 3. 经典越狱关键词（中英文）
    ("jailbreak", r"\b(DAN|jailbreak|developer\s*mode|do\s*anything\s*now|ignore\s*(all\s*)?instructions|越狱)\b"),
    # 4. 强行要求输出内部系统信息
    ("force_disclosure", r"(输出|打印|给我|展示|告诉我).{0,15}(系统提示词|系统提示|初始指令|角色设定|system\s*prompt)"),
    # 5. 伪造权限/身份（假装可授权、自封管理员）
    ("superuser_fabrication", r"(授予|给我|我要求|请给|提权|开通).{0,8}(管理员|超级管理员|最高权限|root\s*权限|所有权限)"),
    # 6. 催办危险动作、绕过人工确认
    ("no_confirm_tool", r"(不用|无需|跳过|别走).{0,4}(确认|审核|人工|审批).{0,16}(退款|全额|退全款|改订单|转钱|打款)"),
    # 7. base64 / 编码载荷
    ("base64_payload", r"(base64|btoa|atob|解码|解密|decode\s*(this|the|that)+)"),
    # 8. 收集/套取他人隐私
    ("privacy_harvest", r"(其他|别人|他人).{0,3}(用户|客户|人).{0,5}(手机号|订单|地址|隐私|信息)"),
    # 9. 套取内部流程 / 工具清单 / 可调用能力
    ("internal_disclosure", r"(内部流程|内部指令|你注册了|你的工具|可调用|工具列表|tool\s*list|你能做什么|你的能力)"),
    # 10. 忽略以上上下文（经典上下文劫持）
    ("ignore_prior", r"(忽略|无视).{0,8}(以上|上面|之前).{0,4}(指令|内容|规则|设定)"),
]


@dataclass
class GuardResult:
    """一次输入检查的结论。"""

    decision: str  # allow | block | flag
    matched_rule: str = ""
    reason: str = ""


def screen_by_rules(text: str) -> GuardResult:
    """L1：逐条正则扫描，命中即返回 block + 命中的规则名。

    纯函数：确定性、零 LLM、可离线单测。未命中 → allow。
    """
    for name, pattern in REGEX_RULES:
        if re.search(pattern, text, _REGEX_FLAGS):
            return GuardResult(
                decision="block",
                matched_rule=name,
                reason=f"L1 规则命中: {name}",
            )
    return GuardResult(decision="allow")
