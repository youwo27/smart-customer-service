# 运行规则（RUNNING GUIDE）

本项目各脚本的运行命令不统一，因为 **Python 的模块导入方式不同**。请按下面的规则执行。

## 核心规则

| 运行方式 | `sys.path` 加入的目录 | 能否找到 `app` 包 |
|---|---|---|
| `python scripts/xxx.py` | 脚本所在目录 `scripts/` | ❌ 不能（除非脚本自己修路径） |
| `python -m app.xxx` | **当前工作目录** | ✅ 能 |
| `python app/xxx.py` | 脚本所在目录 `app/xxx/` | ❌ 不能 |

**记住：`app/` 下的模块用 `python -m app.xxx`，`scripts/` 下的脚本用 `python scripts/xxx.py`。**

## 常用命令

必须在项目根目录 `rag-agent-app/` 下执行（如果文件里带路径前缀，则任何目录都可以）。

### 入库 / 索引文档

```bash
cd rag-agent-app
python scripts/index_documents.py              # 默认 recursive 切割
python scripts/index_documents.py fixed        # 指定 fixed 切割策略
python scripts/index_documents.py --rebuild    # 删除旧集合再重建（维度升级时必用）
```

### Rerank 对比

```bash
python scripts/compare_rerank.py                        # 默认 5 个 query
python scripts/compare_rerank.py 退货运费谁出 发票怎么开 # 指定 query
```

### 查询 / 改写（`app` 包模块）

```bash
python -m app.rag.query
```

> ⚠️ 直接 `python app/rag/query.py` 会报 `ModuleNotFoundError: No module named 'app'`，因为直接运行时 Python 只在脚本所在目录 `app/rag/` 里找模块，找不到上级的 `app` 包。

## 为什么 scripts/ 能用相对路径跑？

`scripts/` 下的脚本（如 `index_documents.py`、`compare_rerank.py`）都在文件顶部手动把**项目根目录**加进了 `sys.path`：

```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
```

所以它们用 `python scripts/xxx.py` 直接跑也能找到 `app` 包。

`app/` 下被当作模块运行的代码（如 `app/rag/query.py`）没有这行，只能依赖 `-m` 把当前工作目录加进搜索路径。

## 给 app/ 下新脚本加路径修正（可选）

如果想让 `app/` 下的脚本也能直接 `python app/xxx/yyy.py` 运行，在文件顶部、`from app...` 导入**之前**加：

```python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
```

注意：`app/rag/` 到项目根是两层 `".."`，而 `scripts/` 到项目根只有一层。
