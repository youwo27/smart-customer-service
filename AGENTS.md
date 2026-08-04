# AGENTS.md — 项目代码规范

> 本文件是 MiniSupport Agent 项目的编码规范，供 AI 编码助手（Claude Code / Cursor / Codex）和所有开发者共同遵守。
>
> **核心原则**：每一个写进代码的 Agent 能力，都必须有日志、有测试、有失败处理。宁可慢，不可糙。

---

## 1. 项目定位

- 名称：`MiniSupport Agent`（电商客服智能系统）
- 形态：FastAPI 异步服务 + SSE 流式 + RAG 知识库 + 工具调用 + 多 Agent 分流
- 技术栈：Python 3.11 / FastAPI / **DeepSeek（OpenAI 兼容协议，可切换 Claude/OpenAI）** / ChromaDB / PostgreSQL / Redis / structlog / LangGraph
- 目标读者（面试叙事）：**从 Demo 思维到生产级系统的完整示例**

---

## 2. 目录结构与职责边界

```
rag-agent-app/
├── app/
│   ├── main.py            # 应用入口：启动/关闭/异常处理，不做业务逻辑
│   ├── config.py          # pydantic-settings 配置，全项目唯一配置源
│   ├── api/               # 接入层：路由定义，只做参数校验与响应返回
│   │   └── routes.py
│   ├── core/              # 核心层：Agent 循环与编排（业务核心，禁止掺入框架代码）
│   │   ├── agent.py           # ReAct AgentLoop
│   │   └── orchestrator.py    # 编排：上下文组装 → Agent → 格式化
│   ├── rag/               # 检索层：RAG 管道
│   │   ├── chunker.py         # 文档切割
│   │   ├── embedder.py        # 向量化
│   │   ├── retriever.py       # 检索（Dense + Sparse + Rerank）
│   │   ├── reranker.py        # 重排序
│   │   └── store.py           # 向量库封装
│   ├── tools/             # 工具层：注册中心 + 内置工具
│   │   ├── registry.py
│   │   └── builtin/           # 查订单 / 查物流 / 申请退款 / 知识检索
│   ├── context/           # 上下文工程：组装 + 压缩
│   │   └── assembler.py
│   └── models/            # 数据模型：Pydantic schema / ORM
│       └── schemas.py
├── tests/                 # 测试：单测（tests/unit/）+ 集成（tests/integration/）
├── docs/                  # 需求分析、架构图、ADR
├── data/                  # 数据：raw_documents（不入库）/ chroma（向量）
├── docker-compose.yml     # 一键启动
└── pyproject.toml         # 依赖 + mypy/ruff/pytest 配置
```

**职责边界铁律**：
1. `api/` 不写业务逻辑（只做请求校验 + 调 orchestrator）
2. `core/` 不依赖 FastAPI（可独立测试的纯逻辑层）
3. `rag/`、`tools/` 只实现自身职责，不直接操作 LLM 客户端（通过注入）
4. 依赖方向单向：`api → core → tools/rag → 基础设施`

---

## 3. 代码硬性规范

### 3.1 配置管理
- **禁止硬编码**任何配置（API Key、模型名、超时时间、URL）
- 所有配置写入 `app/config.py`（pydantic-settings），通过 `.env` 注入
- 新增配置必须同步更新 `.env.example`

### 3.2 日志
- **禁止 `print()` 调试**，一律用 `structlog`
- 每条日志必须携带 `trace_id`、`session_id`、`module`
- 日志级别规范：
  - `DEBUG`：开发调试细节
  - `INFO`：关键流程节点（请求开始/结束、工具调用）
  - `WARNING`：可恢复错误（重试、降级）
  - `ERROR`：需关注（异常、失败）
- 日志中**禁止输出**：完整 API Key、用户敏感信息（手机号、地址、订单号全文）

### 3.3 错误处理
- 所有外部调用（LLM API、数据库、Redis、向量库）**必须有 try-except + 重试逻辑**
- LLM 调用重试：指数退避（base 1s，×3 次），超时后降级
- 工具调用失败：捕获异常 → 将错误信息作为 `tool_result` 返回（让 Agent 自我纠正），不抛给上层
- 允许 Agent 失败的场景，禁止静默吞异常

### 3.4 类型与质量
- 所有函数必须有**完整类型注解**，`mypy --strict` 通过
- `ruff` lint 通过（line-length 100）
- 核心逻辑（AgentLoop / ToolRegistry / chunker / retriever）单元测试覆盖率 **> 60%**
- 每个公共模块必须有 **module docstring**（说明职责 + 关键设计决策）

### 3.5 异步
- 一律 `async def`，禁止 `time.sleep()`（用 `await asyncio.sleep()`）
- 并发控制：`asyncio.Semaphore` 限制最大并发会话
- 所有 IO 操作加超时（`asyncio.wait_for`）

---

## 4. Agent 特有规范

### 4.1 工具设计
- 每个工具：`name` / `description` / `parameters(JSON Schema)` / `async execute()`
- `description` 必须写清：**什么时候用、什么时候不用、参数怎么写**（决定工具调用准确率）
- 危险工具（退款、改地址）必须声明 `required_permissions`，运行时校验 + 二次确认
- 工具返回**结构化结果**，禁止返回大段无格式文本

### 4.2 上下文
- 系统提示词分层：System（固定）→ Task（本次任务）→ Dynamic（动态信息）
- 上下文窗口预算：System 10K + 工具定义 20K + 对话历史 150K + 预留 20K
- 超过阈值必须触发压缩（摘要历史），禁止暴力截断

### 4.3 安全
- 用户输入 → 输入过滤（正则 + LLM 分类）→ Agent → 输出审核 → 用户
- 高危操作（退款 > ¥500 / 赔偿承诺）→ 强制 HITL 人工审批
- 日志中的 PII 自动脱敏

---

## 5. 测试规范

- 测试文件命名：`test_<模块名>.py`，与源码目录镜像
- 单元测试：AgentLoop 终止条件 / 工具解析 / 重试逻辑 / chunker 边界
- 集成测试：`/chat` 端到端、检索质量
- 每个 PR（或每个 Day 里程碑）必须：`pytest && mypy && ruff`

---

## 6. Git 规范

- 提交信息格式：`<type>(<scope>): <summary>`
  - `type`：feat / fix / refactor / docs / test / chore
  - 例：`feat(tools): 添加 query_order 工具`
- 每个功能模块一个 commit，不攒一堆文件一次提交
- 提交前跑通测试

---

## 7. 对 AI 助手的要求（Claude Code / Cursor / Codex）

- 收到任务后，先读 `docs/REQUIREMENTS.md`（如果有）再动手
- 修改前先看现有代码风格，**沿用**而非另起炉灶
- 完成功能后，主动补充对应测试和日志
- 不确定的需求，先列出待确认问题，再写代码
- 禁止"能跑就行"：完成标准 = 测试通过 + 日志完整 + 失败有兜底
