"""安全防护层（Day 8）— 纵深防御的输入/动作/输出三道闸。

包内模块：
- input_guard.py    L1 正则规则 + GuardResult（确定性阻断）
- classifier.py     L2 LLM 语义三分类（normal/suspicious/attack），独立超时
- guardrails.py     InputGuard：L1 先跑、L2 兜底的组合管道
- permissions.py    （Day 8 Part 4）工具动作权限，危险操作默认拒绝
- audit.py          （Day 8 Part 4）审计日志
- proxy.py          （Day 8 Part 4）受控工具注册器
- pii.py            （Day 8 Part 5）PII 检测脱敏
"""
