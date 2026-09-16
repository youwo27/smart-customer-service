# MiniSupport Agent — 电商客服智能系统

> 集成 **RAG → Agentic RAG → 多 Agent 分流 → 安全防护 → 可观测性** 的电商客服 Agent，从 Demo 思维到生产级系统的完整示例。

## 快速开始

```bash
# 1. 安装依赖（Python >= 3.11）
pip install -e ".[dev]"

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，至少填 LLM_API_KEY（默认 DeepSeek，OpenAI 兼容协议）

# 3. 起依赖（Postgres / Redis / Chroma / Jaeger，不需要可只起 jaeger）
docker compose up -d

# 4. 索引知识库文档（首次必做，写入 data/chroma）
python scripts/index_documents.py

# 5. 启动服务
uvicorn app.main:app --reload
```

验证：

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "怎么退货？"}'
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message": "查一下订单 12345 的物流"}'
```

- API 文档：`GET http://localhost:8000/docs`
- Trace 面板：`GET http://localhost:16686`（Jaeger UI，service 选 `minisupport-agent`）

> ⚠️ 不看 trace 就把 `OTEL_ENABLED=false`：Jaeger 没起时 OTel SDK 会按批重试导出，往日志里刷 `Transient error ... UNAVAILABLE`（不影响业务，但很吵）。
> ⚠️ `LOG_SAMPLE_RATE` 默认 `0.1`，本地调试想看清全量 INFO 日志设成 `1.0`；WARNING/ERROR 与安全审计事件不受采样影响。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/v1/chat` | 一次性返回完整回复（`session_id` 可选，用于多轮） |
| POST | `/api/v1/chat/stream` | SSE 流式：按 `tool_call` / `tool_result` / `assistant` / `end` 逐帧下发 |

## 架构

```
接入层(FastAPI + OTel 中间件)
  → 编排层(Orchestrator：输入守卫 → 上下文组装 → AgentLoop → 出口脱敏/落库)
    → Agent 层(AgentLoop：ReAct 循环 + 工具超时/LLM 重试)
      ├── 检索层(RAG：chunker / hybrid_search(dense+sparse) / reranker)
      ├── 工具层(受控注册器：权限校验 + 审计 → 6 个内置工具)
      └── 安全层(InputGuard L1 正则 + L2 语义分类 / PermissionGuard / 审计 / PII 脱敏)
  → 基础设施层(DeepSeek·Zhipu API / PostgreSQL / Redis / ChromaDB / Jaeger)
```

关键设计：安全层用 `SecuredToolRegistry` 包住真注册器，Agent 完全无感；追踪只埋在**唯一关口**（orchestrator.handle / AgentLoop.run / LLMClient / Retriever / Reranker / 工具执行），不散落在业务代码里。

## 目录

```
app/
├── main.py            # 入口：装配编排器 + 追踪中间件 + lifespan
├── config.py          # 配置（pydantic-settings）
├── logging_config.py  # structlog：trace/session 上下文注入 + 采样 + 脱敏
├── api/routes.py      # 路由（仅校验/返回，无业务逻辑）
├── core/              # AgentLoop（ReAct 循环）+ Orchestrator
├── context/           # 上下文组装 + 超长压缩
├── rag/               # loader / chunker / embedder / store / retriever / reranker / agentic
├── tools/             # 注册中心 + 6 个内置工具（3 RAG + 3 客服业务）
├── graph/             # LangGraph 对照实现（多 Agent 分流 + Agentic RAG 状态图）
├── security/          # 输入守卫 / 权限 / 审计 / PII / 出口守卫
├── observability/     # OTel 装配 + span() 便捷封装
├── storage/           # 会话持久化（memory / sqlite）
└── models/schemas.py  # 数据模型
```

## 运行脚本

脚本运行方式不统一（`app/` 下用 `python -m`，`scripts/` 下直接 `python`），规则见 [RUNNING.md](RUNNING.md)。

| 脚本 | 用途 |
|------|------|
| `scripts/index_documents.py` | 文档入库（`fixed` 可指定切割策略，`--rebuild` 重建集合） |
| `scripts/compare_rerank.py` | 有无 Rerank 的效果对比 |
| `scripts/compare_dense_vs_hybrid.py` | 稠密 vs 混合检索对比 |
| `scripts/evaluate_agentic.py` | Agentic RAG 效果评估 |
| `scripts/demo_agentloop.py` / `demo_react.py` | 自研 Agent 循环 Demo |
| `scripts/demo_langchain.py` / `lcel_rag_chain.py` | LangChain 对照实现 |
| `scripts/demo_langgraph.py` | LangGraph 多 Agent 分流 Demo |

## 测试

```bash
pytest                      # 17 个测试文件：Agent 循环 / RAG / API / 安全 / 可观测性
ruff check . && mypy app    # mypy 全仓基线在 Day1–5 遗留模块上仍红，app/graph 与新增模块干净
```

## 文档

- [需求分析单](docs/requirements.md)
- [代码规范](AGENTS.md)
- [运行规则](RUNNING.md)
- [14 天学习计划](../Agent两周速成计划.md)

## 状态

| Day | 内容 | 状态 |
|-----|------|------|
| Day 1 | 架构 + 项目骨架 | ✅ |
| Day 2 | RAG：切割 + Embedding + 存储 | ✅ |
| Day 3 | RAG：混合检索 + Rerank + Agentic | ✅ |
| Day 4 | Agent 循环 + 工具系统 | ✅ |
| Day 5 | LangChain 框架实战与对比 | ✅ |
| Day 6 | LangGraph 与多 Agent | ✅ |
| Day 7 | 编排层 + 上下文组装 + 会话持久化 + SSE | ✅ |
| Day 8 | 安全防护体系（输入守卫 / 权限 / 审计 / PII） | ✅ |
| Day 9 | 可观测性(上)：日志上下文 + 采样 + 出口脱敏 + OTel 追踪 | 🟡 Part 4/7 未完（`app/rag/query.py` 裸 print 未清；真实 OTLP→Jaeger 未验证） |
| Day 10 | 可观测性(下)：指标 + 告警 | ⬜ 计划文档已生成，未实现 |
