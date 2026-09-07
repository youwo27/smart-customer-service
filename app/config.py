"""应用配置管理 — 所有配置通过环境变量注入，零硬编码。

规则（见 AGENTS.md 3.1）：
- 禁止在业务代码里写死任何配置
- 新增配置必须同步更新 `.env.example`
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置，自动从 .env 文件和环境变量加载。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # === LLM ===
    # provider 可切换：deepseek | anthropic | openai（DeepSeek 走 OpenAI 兼容协议）
    llm_provider: str = "deepseek"
    llm_api_key: str
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"  # 通用对话（DeepSeek-V3.x）
    llm_fast_model: str = "deepseek-chat"  # 简单/快速任务；复杂推理可换 deepseek-reasoner
    # 兼容字段：切回 Anthropic 时使用（默认空，不干扰 DeepSeek）
    anthropic_api_key: str = ""

    # === Embedding ===
    # provider：chroma（内置，离线）| zhipu（智谱 embedding-3，中文好）| siliconflow（bge）
    embedding_provider: str = "zhipu"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    embedding_model: str = "embedding-3"

    # === Rerank ===
    # provider：""（跳过重排）| siliconflow（bge-reranker）
    rerank_provider: str = ""
    rerank_api_key: str = ""
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # === Database ===
    database_url: str = "postgresql+asyncpg://agent:agent123@localhost:5432/agent_db"
    redis_url: str = "redis://localhost:6379/0"

    # === ChromaDB ===
    chroma_persist_dir: str = "./data/chroma"

    # === Server ===
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # === Agent ===
    max_turns: int = 15
    tool_timeout_seconds: int = 30
    llm_retry_max: int = 3
    llm_retry_base_delay: float = 1.0

    # === Context ===
    max_context_tokens: int = 180_000
    compression_threshold_tokens: int = 150_000
    compression_keep_recent: int = 8  # 压缩时保留的最近消息条数（再早的进摘要）

    # === Session（会话持久化，Day 7） ===
    session_store_backend: str = "memory"  # memory（开发/测试） | sqlite（重启恢复）
    session_store_path: str = "./data/sessions.db"


# 全局单例（整个项目 import 这一个对象即可）
settings = Settings()
