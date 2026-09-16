"""可观测性层（Day 9）— 让 Agent 的每一步"看得见"。

包内模块：
- tracing.py  OpenTelemetry 全链路追踪：TracerProvider 装配 + `span()` 埋点助手

与 logging_config 的分工：日志回答"发生了什么"（离散事件、带 trace_id/session_id），
追踪回答"这一步花了多久、挂在谁下面"（有起止时间的树）。两者用同一套上下文串起来。
"""
