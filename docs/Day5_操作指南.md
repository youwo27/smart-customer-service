# Day 5 操作指南：把 Day 4 的 Agent 用 LangChain 重写一遍（照着做就行）

> **这个文档是给谁看的**：看不懂《Day5_LangChain框架实战与对比.md》里那些术语（StructuredTool、coroutine、LCEL…）的人。这篇只讲"你**按什么顺序、敲什么命令、看到什么算成功**"，概念只在需要的地方用一句人话带过。
>
> **前提**：你的项目在 `rag-agent-app/`，Day 4 自研版已跑通，`.env` 里有 DeepSeek key。

---

## 0. 一句话讲清楚今天要干嘛

你 Day 4 手写了一个 Agent（自己管循环、自己管消息、自己调工具）。今天用 LangChain 框架**把同一套东西重写一遍**，然后对比：框架替你省了什么、代价是什么。

**你不写新代码**——文件已经帮你建好了。你只需要**跑命令、看输出**，验证每条链路都通。

---

## 1. 先跑通 3 个脚本（20 分钟）

**所有命令都在项目根目录** `rag-agent-app/` 下执行（先 `cd rag-agent-app`）。

### ① 自研版 —— 对照组（看一遍就行）

```bash
python scripts/demo_agentloop.py "退货政策是什么？"
```

你会在终端看到：Agent 调用了 `search_knowledge_base` 工具 → 拿到知识库片段 → 回答。

**成功标准**：最后有一段"退货政策"的回答，中间有 `[tool_calls] search_knowledge_base(...)` 这样的行。
 [tool_calls] search_knowledge_base({"query": "退货政策"})
  [tool] {"chunks": [{"id": "01-return-policy-chunk-0", "text": "# 退货政策\n\n## 退货条件\n\n- 支持 7 天无理由退货（签收后 7 天内），从签收次日起计算\n- 商品需保持完好：未使用、未清洗、包装吊牌齐全\n- 定制类商品、贴身衣物（内衣/袜子/泳装）、
  [assistant] 为您介绍我们的退货政策：

**📦 退货时效**
- **无理由退货**：签收后 **7 天内**可申请（从签收次日起算）
- **质量问题退货**：签收后 **15 天内**可申请
- 超过时效请先联系人工客服说明情况

**✅ 退货条件**
- 商品需保持完好：未使用、未清洗、包装吊牌齐全
- 定制类商品、贴身衣物

---- 最终回答 ----
为您介绍我们的退货政策：

**📦 退货时效**
- **无理由退货**：签收后 **7 天内**可申请（从签收次日起算）
- **质量问题退货**：签收后 **15 天内**可申请
- 超过时效请先联系人工客服说明情况

**✅ 退货条件**
- 商品需保持完好：未使用、未清洗、包装吊牌齐全
- 定制类商品、贴身衣物（内衣/袜子/泳装）、生鲜食品不支持无理由退货
- 明确标注"不支持退换"的特价商品除外

**💰 退货运费**
- 无理由退货：运费由买家承担（有运费险则由保险公司赔付）
- 质量问题退货：运费由商家承担（先垫付，退款时一并退还）
- 换货：往返运费由商家承担

**📝 退货流程**
1. 订单详情页点击"申请退货"
2. 填写原因并上传商品照片
3. 商家 1 个工作日内审核
4. 审核通过后 7 天内寄回并填写物流单号
5. 仓库收货验货（1-2 个工作日）
6. 验货通过后退款原路返回（1-3 个工作日到账）

请问您需要办理退货吗？如果有具体订单，可以告诉我订单号帮您查看哦～

> 这一版是"自己造的轮子"，跑它只是为了有对照。下面②才是今天的主角。

### ② LangChain 版 —— 今天的主角

```bash
python scripts/demo_langchain.py "退货政策是什么？"
```

看到的关键输出：

```
---- Agent 调用过程（intermediate_steps） ----
  [tool] search_knowledge_base({'query': '退货政策'}) → {'chunks': [...]}
---- 最终回答 ----
[一段退货政策的回答]
```
[tool] search_knowledge_base({'query': '退货政策'}) → {'chunks': [{'id': '01-return-policy-chunk-0', 'text': '# 退货政策\n\n## 退货条件\n\n- 支持 7 天无理由退货（签收后 7 天内），从签收次日起计算\n- 商品需保持完好：未使用、未清洗、包装吊牌齐全\n- 定制类商品、贴身衣物（内衣/袜子/泳装）、

---- 最终回答 ----
为您整理了我们的退货政策，主要如下：

**📦 退货条件**
- **7天无理由退货**：签收后7天内（从签收次日起算）
- 商品需保持完好：未使用、未清洗、包装吊牌齐全
- 定制类商品、贴身衣物（内衣/袜子/泳装）、生鲜食品不支持无理由退货
- 促销特价商品（明确标注"不支持退换"）除外

**⏰ 退货时效**
- 无理由退货：签收后7天内申请
- 质量问题退货：签收后15天内申请
- 超过时效无法在线申请，可联系人工客服说明情况

**💰 退货运费**
- 无理由退货：运费由买家承担（有运费险由保险公司赔付）
- 质量问题退货：运费由商家承担（买家先垫付，退款时一并退还）
- 换货：往返运费由商家承担

**📋 退货流程**
1. 订单详情页点击"申请退货"
2. 填写原因、上传商品照片
3. 商家1个工作日内审核
4. 审核通过后7天内寄回并填写物流单号
5. 仓库收货验货（1-2个工作日）
6. 验货通过后退款原路返回（1-3个工作日到账）

请问您是想了解退货的哪方面，还是有具体的订单需要处理呢？😊

**成功标准**：和①用了**同一个工具**、答得**差不多**。这就对了——说明换框架没改变 Agent 行为。

再试另外两条路由：

```bash
python scripts/demo_langchain.py "订单 12345 到哪了？"
python scripts/demo_langchain.py "我要退款"
```

预期：
- 第二条 → `[tool] query_order(...)` → 已发货
- 第三条 → **不调 apply_refund**，而是追问"哪个订单？退款原因？"（这是对的！危险操作要守门）

---

## 2. 把三个脚本的参数逻辑搞清楚（10 分钟，看文件）

看不懂没关系，看这三个文件各自"入口"长什么样就行：

| 文件 | 它是干嘛的 | 对应自研哪个文件 |
|---|---|---|
| `scripts/_lc_tools.py` | **工具转换器**：把自研 6 个工具包装成 LangChain 认识的 `StructuredTool` | 自研的 `app/tools/registry.py` |
| `scripts/demo_langchain.py` | LangChain 版 Agent 主脚本 | 自研的 `scripts/demo_agentloop.py` |
| `scripts/lcel_rag_chain.py` | 一条 RAG 问答链（检索→拼上下文→回答） | 自研的"检索→组装→生成"流程 |

**用不用看代码？** 先不用。等你跑完所有命令、有感觉了，再回头读 `_lc_tools.py`（注释里把每一步"为什么"都写了）。

---

## 3. 跑 LCEL RAG 链（10 分钟）

```bash
python scripts/lcel_rag_chain.py "退货运费谁出？"
```

你会看到：**不调任何工具**，直接把知识库内容整理成回答。因为这是一条"纯问答链"，不是 Agent（没有工具调用那一步）。

**成功标准**：输出"无理由退货运费买家承担 / 质量问题商家承担"这类基于知识库的答案。

---

## 4. 跑测试（5 分钟，不花钱——用假数据，不真调 DeepSeek）

```bash
python -m pytest tests/test_langchain_adapters.py -q
```

**成功标准**：`13 passed`。这 13 个测试验证了：
- 6 个工具都正确转换成了 `StructuredTool`（名字/描述没丢）
- 工具能真的执行（`query_order` 返回 mock 订单）
- LCEL 链用假 LLM 能跑通
- Agent 能正常组装

> 全量测试再跑一遍确认没破坏旧功能：
> ```bash
> python -m pytest -q   # 期望 51 passed
> ```

---

## 5. 把术语翻译成人话（现在你该能看懂了）

| 术语 | 人话解释 |
|---|---|
| `StructuredTool` | 一张"工具名片"，上面写清楚工具叫什么、干嘛用、需要哪些参数。LLM 看完名片才会正确调用它 |
| `coroutine=` | 你的工具是 `async def` 的（异步），LangChain 要求异步函数用 `coroutine=` 传；同步函数才用 `func=`。**你用异步，就用 `coroutine=`** |
| `from_function` 自动生成 schema | 你不用手写参数说明。函数签名里 `query: str, top_k: int = 5` 这行注解，LangChain 自动变成 LLM 能懂的 JSON 格式 |
| `create_tool_calling_agent` | 一句话让框架帮你建好 Agent：给它 LLM + 工具列表 + 提示词，它内部自动处理"调用工具→拿结果→再回答"的循环 |
| `AgentExecutor` | Agent 的"跑步机"：负责真的跑起来、循环、出错兜底 |
| LCEL / `\|` | 把检索→提示→LLM→输出**串成一条流水线**，`\|` 就是把上一步输出交给下一步 |

**对照表**（这就是《对比文档》里"框架替我省了什么"的逐条对应）：

| 自研（Day 4） | LangChain（Day 5） |
|---|---|
| 自己写 `while` 循环控制轮数 | `AgentExecutor` 内建循环 |
| 自己组装 assistant/tool 消息 | `ChatPromptTemplate` 的 placeholder 占位自动维护 |
| 自己解析 `tool_calls` | `create_tool_calling_agent` 内部处理 |
| 自己写重试 + 异常兜底 | `handle_parsing_errors=True` 一行搞定 |

---

## 6. 你接下来要做的"动脑"部分（可选，面试用）

跑完上面，你已经完成了 Day 5 的"动手"部分。如果还想深入，打开这两个文件各读 10 分钟：

1. **`scripts/_lc_tools.py`** —— 看 `make_lc_tool()`：它是怎么把自研工具"翻译"成 LangChain 工具的（注释写了每一步为什么）。

2. **`docs/Day5_framework_compare.md`** —— 这是 Day 5 的"作业答案"，面试就问这里面的结论。

面试能说的一句话：**"我用 LangChain 把自研 Agent 重写了一遍，同样 3 条路由结果一致，但代码从 160 行降到 30 行；代价是自定义循环逻辑和精确 debug 变难，而且 0.3 和 1.x 版本 API 不兼容。"**

---

## 7. 常见坑（碰到了再来看）

1. **`import langchain.agents` 报 `ModuleNotFoundError: langchain_core.memory`** → 版本装错了。`langchain 0.3.x` 必须配 `langchain-core <1.0.0`。项目已锁好版本，别手动 `pip install langchain-core`（会装回 1.x）。
2. **中文乱码（�）** → Windows 终端编码问题，不影响逻辑。脚本顶部已加 UTF-8 处理；终端里再执行 `chcp 65001` 更稳。
3. **`TypeError: unsupported operand | for 'function'`** → 普通函数不能直接 `\|`，要用 `RunnableLambda` 包一层（`lcel_rag_chain.py` 里已示范）。
4. **跑 demo 报 key 错误** → 检查 `rag-agent-app/.env` 里 `LLM_API_KEY` 是否填了 DeepSeek key。

---

## 附：一条命令检查全绿

```bash
cd rag-agent-app
python -m pytest -q && python -m mypy scripts tests && python -m ruff check scripts tests app
```

> 注：`mypy app` 里 Day 2/3 遗留的 `store.py` 等报错**不是** Day 5 引入的，暂不影响本日交付。
