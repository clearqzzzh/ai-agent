"""
配置模块 — 对标 Spring Boot application.yml
所有环境相关配置集中管理，启动时一次性加载。
"""

import os
from pathlib import Path

from loguru import logger

# ── .env 文件加载 ───────────────────────────────────
def _load_dotenv(path: Path | None = None) -> None:
    env_file = path or (Path(__file__).resolve().parent.parent / ".env")
    if not env_file.exists():
        logger.info(f"未找到 .env 文件: {env_file}")
        return
    loaded = 0
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if not key:
                continue
            os.environ[key] = val
            loaded += 1
    logger.info(f"从 {env_file.name} 加载了 {loaded} 个配置项")


_load_dotenv()

# ── DeepSeek ─────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-your-key")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ── 千问（视觉识别）─────────────────────────────────
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-vl-plus")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# ── Redis ────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_SESSION_TTL = 60 * 60 * 24  # 会话 24 小时后自动过期

# ── 上下文管理 ───────────────────────────────────────
MAX_HISTORY_TOKENS = 12_000
RECENT_WINDOW_TOKENS = 4_000
SUMMARY_MAX_TOKENS = 500
MAX_ROUNDS = 10  # 每会话最多保留 10 轮历史对话
SESSION_DIR = Path(__file__).resolve().parent.parent / "sessions"  # 旧数据，迁移后可删

# ── 商城 API ─────────────────────────────────────────
PRODUCT_API_URL = "http://localhost:8082/api/product/search"
PRODUCT_LOGIN_URL = "http://localhost:8083/login/accountLogin"
PRODUCT_API_USERNAME = os.getenv("PRODUCT_API_USERNAME", "")
PRODUCT_API_PASSWORD = os.getenv("PRODUCT_API_PASSWORD", "")
PRODUCT_API_TOKEN = os.getenv("PRODUCT_API_TOKEN", "")

# ── 订单服务 API ─────────────────────────────────────
ORDER_SERVICE_URL = os.getenv("ORDER_SERVICE_URL", "http://localhost:8084")
