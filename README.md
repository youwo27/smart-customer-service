# MiniSupport Agent — 电商客服智能系统

> 集成 **RAG → Agentic RAG → 多 Agent 分流 → 人在回路** 的电商客服 Agent，从 Demo 思维到生产级系统的完整示例。

## 快速开始

```bash
# 1. 安装依赖
pip install -e ".[dev]"

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 ANTHROPIC_API_KEY

# 3. 启动服务
uvicorn app.main:app --reload
```

验证：

- 健康检查：`GET http://localhost:8000/health`
- API 文档：`GET http://localhost:8000/docs`
- 聊天：`POST /api/v1/chat` `{"message": "怎么退货？"}`

## 架构

```
接入层(FastAPI) → 编排层(Orchestrator) → Agent层(AgentLoop)
                                        ├── 检索层(RAG：chunker/retriever/reranker)
                                        └── 工具层(ToolRegistry：query_order/query_logistics/apply_refund)
→ 基础设施层(PostgreSQL / Redis / ChromaDB / LLM Client)
```

## 目录

```
app/
├── main.py            # 入口
├── config.py          # 配置（pydantic-settings）
├── api/routes.py      # 路由
├── core/              # AgentLoop + Orchestrator
├── rag/               # RAG 管道
├── tools/             # 工具注册中心 + 内置工具
├── context/           # 上下文组装 + 压缩
└── models/schemas.py  # 数据模型
```

## 文档

- [需求分析单](docs/REQUIREMENTS.md)
- [代码规范](AGENTS.md)
- [14 天学习计划](../Agent两周速成计划.md)

## 状态

| Day | 内容 | 状态 |
|-----|------|------|
| Day 1 | 架构 + 项目骨架 | ✅ |
| Day 2 | RAG：切割 + Embedding + 存储 | ⬜ |
| Day 3 | RAG：混合检索 + Rerank + Agentic | ⬜ |
| Day 4 | Agent 循环 + 工具系统 | ⬜ |
