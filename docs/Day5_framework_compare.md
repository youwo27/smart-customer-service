# Day 5 框架对比：自研（Day 4）vs LangChain（0.3.26）

> **一句话定位**：LangChain 是"通用 Agent 样板的封装"，自研是"业务决策的全部掌控"。Day 5 用同一份能力跑两条路线，**用实测数据而不是口说**比较五个维度，最后给出选型结论。
>
> **实测日期**：2026-08-09 ｜ **实测环境**：langchain 0.3.26 / langchain-core 0.3.86 / langchain-openai 0.3.28 / DeepSeek chat

---

## 0. 先记录一个环境坑（踩过才有说服力）

开工第一天先撞上**依赖版本不兼容**：

```
langchain 0.3.26 has requirement langchain-core<1.0.0,>=0.3.66, but you have langchain-core 1.5.3.
```

`langchain-core 1.5.3`（1.x 线）删掉了 `langchain_core.memory`，导致 `import langchain.agents` 直接 `ModuleNotFoundError`。修复：把 `langchain-core` 钉回 `0.3.86`、`langchain-anthropic` 钉回 `0.3.22`，并在 `pyproject.toml` 里加 `langchain-core<1.0.0` 的版本上限，防止 pip 再装回 1.x。

**面试点**：框架版本割裂是 LangChain 出了名的痛点（0.3 和 0.4/1.x 的 Agent API 完全不兼容，`create_agent` 和 `create_tool_calling_agent` 是两套东西）。项目里锁版本是选 LangChain 的必修课。

---

## 1. 实测对比基准（先跑数据，再写结论）

同一份能力两条实现路线，跑同样 3 条路由 query（都是 DeepSeek）：

| 用户输入 | 自研（demo_agentloop.py） | LangChain（demo_langchain.py） | 结论 |
|---|---|---|---|
| 退货政策是什么？ | search_knowledge_base → 按知识库回答 | search_knowledge_base → 按知识库回答 | ✅ 路由一致 |
| 订单 12345 到哪了？ | query_order → 已发货 | query_order → 已发货 | ✅ 路由一致 |
| 我要退款 | 不给足订单号/原因时不调 apply_refund | 同样不调 apply_refund，主动要订单号 | ✅ 危险操作守门一致 |

两条路线对 3 条 query 的路由结果**完全一致**——这是"框架没改变 Agent 行为"的基线证据，后面比的是"实现差多少"。

> 补充：LangChain 版对"我要退款"不调 `apply_refund`，靠的是工具 description 里"仅当用户明确要求且提供订单号、金额、原因时调用"这段提示。和自研一样，**工具 description 的质量决定路由准确率**——这是框架替不了的核心价值。

---

## 2. 五维对比（有代码引用、有实测）

| 维度 | 自研（Day 4） | LangChain（Day 5） | 我的结论 |
|---|---|---|---|
| **开发效率** | 循环+消息+重试+统计共 3 文件（agent.py 162 行 / registry.py 76 行 / client.py 222 行） | Agent 组装约 30 行（demo_langchain.py） | 从零到跑通：自研花了大半个 Day 4，LangChain 一个上午。但自研的 162 行里有 60% 是"安全 + 统计 + 兜底"，框架省掉的是这 60% |
| **灵活性** | while 循环自己改（改终止条件 = 改 for range） | 改终止/降级逻辑要绕过框架（intermediate_steps 拿不到的部分藏在框架内部） | "单轮最多调 2 个工具"：自研改 `range(max_turns)` 就完事；LangChain 要在 AgentExecutor 的 `max_iterations` + 自定义 `AgentStoppingCond` 之间绕。**改不动的是框架，改得动的是自研** |
| **调试体验** | 报错栈直指自己代码（DeepSeek 原始响应 `_parse` 就在眼前） | 出错要查框架层（模型发了错格式 tool_call，先看到 LangChain 包装栈） | 实测遇到 pydantic `ValidationError`（见下），看到的是框架抛的，不是业务错误。自研版同样情况返回 `{"error": ...}` 让 Agent 自我纠正 |
| **学习曲线** | 无新抽象（字典/函数/循环） | Runnable / Tool / ChatPromptTemplate / AgentExecutor 四层新概念 | Runnable 的 `\|` pipe 是函数式 `compose` 的语法糖，理解后确实优雅；但"为什么 `.invoke/.ainvoke/.stream` 四件套"要专门学 |
| **生产就绪度** | 日志/统计自己写（structlog + token 累加） | 回调/追踪开箱即用（callbacks 挂 token 统计、LangSmith 追踪） | LangChain 开箱即用是真的：LCEL 链天然 `.stream()`（Day 7 SSE 直接复用）；自研要自己加回调接口 |

### 实测案例：缺参数的 apply_refund，两边各抛什么？

这是这次实测里最有意思的一个差异：

```python
# 自研版：execute(**kwargs) 拿到缺参 → 返回结构化错误，Agent 能自我纠正
await QueryOrderTool().execute(order_id="12345")          # {"error": "..."}

# LangChain 版：StructuredTool 先做 pydantic 必填校验 → 直接抛 ValidationError
await apply_refund.ainvoke({"order_id": "12345", "amount": 899})
# pydantic.ValidationError: reason Field required
```

自研版把"参数缺失"当成**业务错误**交给 Agent 纠正；LangChain 的 `args_schema` 把"参数缺失"当成**框架异常**拦在函数外。框架的"替你校验"在这里变成了调试时的额外心智负担——**两种错误哲学，没有对错，但要心里有数**。

---

## 3. LangChain 帮你做了什么（逐条对应自研代码）

1. **消息管理**：`agent_scratchpad` 占位 + `format_to_tool_messages` 自动维护 assistant 的 tool_call + tool 结果。自研的 `_assistant_msg` / `_tool_msg`（[agent.py:77-102](app/core/agent.py#L77-L102)）不用写了。
2. **循环控制**：终止条件、`handle_parsing_errors`、`max_iterations` 内建。自研的 `@retry` + `max_turns` + `_execute_tool_safely` 兜底（[agent.py:104-118](app/core/agent.py#L104-L118)）不用写了。
3. **Provider 抽象**：同一个 `create_tool_calling_agent`，换 provider 只改一行（`ChatOpenAI` → `ChatAnthropic`），工具定义和 Agent 组装完全不动。自研版要写一个 `AnthropicClient`（[anthropic_client.py](app/llm/anthropic_client.py) 那个 `NotImplementedError` 就是欠的账）。
4. **统一接口**：`invoke / ainvoke / stream / bind` 四件套——同步、异步、流式同一套 API。自研的 `chat_with_tools` 只有 async 一种。

**但代价**：
- 自定义逻辑要绕过框架抽象（并行工具失败的降级策略、精确 token 控制）。
- Debug 时模型发了错格式的 tool_call，你看到的是框架报错栈，不是 DeepSeek 原始响应。
- **版本割裂风险**：0.3 ↔ 0.4/1.x API 不兼容，升级可能牵一发动全身（见 §0）。

---

## 4. Provider 切换对照实验（10 分钟）

LangChain 版切 Claude 只需改 `build_llm()` 一处：

```python
# DeepSeek（langchain-openai）
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="deepseek-chat", openai_api_key=..., openai_api_base=...)

# → Claude（langchain-anthropic），工具定义和 Agent 组装一行不改
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-5", anthropic_api_key=...)
```

自研版做同样的事要写整套 `AnthropicClient`（协议不同：Anthropic 的 tool_use 消息格式和 OpenAI 兼容协议不一样）。**这是"框架抽象掉 provider 差异"最直接的证据**——但反过来说，如果业务只有 DeepSeek 一个 provider，这套抽象就是纯成本。

---

## 5. 选型结论

> **什么时候用 LangChain**：
> - 快速原型 / PoC，先跑通再说
> - 要 provider 切换（DeepSeek ↔ Claude 一行换）
> - 要流式 / 回调追踪开箱即用（`.stream()`、callbacks、LangSmith）
> - 团队已经用 LangChain，维护成本共享
>
> **什么时候自己写**：
> - 核心循环逻辑要深度定制（终止条件、降级策略、并行失败处理）
> - 需要精确控制消息和 token（自研每轮手动 append，看得见每个字节）
> - 调试要求看见 DeepSeek 原始响应（框架把错误包装了一层）
> - 长期维护要避免框架升级破坏（0.3 ↔ 1.x API 割裂）
>
> **我们的项目为什么自研 + LangGraph**：
> 核心逻辑（Agent 循环、消息、token、工具守门）需要完全掌控，所以自研做底座；
> 多 Agent 编排层用 LangGraph（Day 6）——框架只做"编排"，不做"业务决策"。

**一句话总结**：框架省的是"通用样板"（消息管理、循环、provider 抽象），你的核心价值是"业务决策"（工具 description 写多准、危险操作守多严、降级策略怎么定）——这些框架替不了你。**先跑通一个工具，再加剩下的。**

---

## 附：实测数据速查

| 指标 | 自研 | LangChain |
|---|---|---|
| Agent 核心代码量 | agent.py 162 行 + registry.py 76 行 | 组装约 30 行 + _lc_tools.py 转换层 |
| 工具 schema | 手写 JSON Schema（parameters 字段） | 从类型注解/args_schema 自动生成（本实测验证明) |
| 消息维护 | 手动 `_assistant_msg` / `_tool_msg` | `agent_scratchpad` 占位自动管理 |
| token 统计 | 手动累加 `usage.input_tokens + output_tokens` | 需自挂 callback（默认不给你） |
| 参数缺省处理 | 业务错误 `{"error": ...}` | pydantic `ValidationError` |
| 单测（不真调 API） | 19 个全绿（Day 4） | 本日 13 个全绿（test_langchain_adapters.py） |
