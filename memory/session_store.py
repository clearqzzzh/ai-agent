"""
会话存储 — Redis 生产级方案
============================
  - Redis Hash：agent:session:{id}  → messages + summary + user_id
  - Redis ZSet：agent:sessions:list → session_id 按时间排序（用于列表查询）
  - 历史轮数上限：MAX_ROUNDS（默认10轮=20条消息）
  - TTL 24h 自动清理
"""

import json

import redis
from loguru import logger

from config import REDIS_URL, REDIS_SESSION_TTL, MAX_ROUNDS

# ── Redis 连接 ─────────────────────────────────────────
_pool: redis.ConnectionPool | None = None


def _r() -> redis.Redis:
    """获取 Redis 客户端。连接失败抛异常，调用方自己决定降级。"""
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool.from_url(
            REDIS_URL,
            socket_connect_timeout=2, socket_timeout=3,
            protocol=2,
        )
    return redis.Redis(connection_pool=_pool)


def _key(session_id: str) -> str:
    return f"agent:session:{session_id}"


# ── 会话索引（Redis ZSet，按时间排序）──
_SESSIONS_INDEX = "agent:sessions:list"


def _touch_index(session_id: str):
    """更新会话索引的时间戳"""
    import time
    try:
        _r().zadd(_SESSIONS_INDEX, {session_id: time.time()})
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════
#  对外 API
# ═══════════════════════════════════════════════════════════

def get_or_create_session(session_id: str) -> dict:
    """从 Redis 读取会话。Redis 不可用返回空会话。"""
    try:
        r = _r()
        data = r.hgetall(_key(session_id))
        if data:
            messages = json.loads(data.get(b"messages", b"[]"))
            summary = data.get(b"summary", b"").decode("utf-8")
            user_id_raw = data.get(b"user_id", b"")
            user_id = int(user_id_raw) if user_id_raw else None
            logger.debug(f"会话 {session_id[:8]} 从 Redis 恢复（{len(messages)}条）")
            return {"messages": messages, "summary": summary, "user_id": user_id}
    except Exception as e:
        logger.warning(f"Redis 读取失败，返回空会话: {e}")
    return {"messages": [], "summary": "", "user_id": None}


def save_session(session_id: str, messages: list[dict], summary: str,
                 user_id: int | None = None) -> None:
    """保存到 Redis，自动截断超过 MAX_ROUNDS 轮的消息。"""
    # 裁剪：只保留最近 MAX_ROUNDS 轮（一轮 = 一次 user + 一次 assistant）
    trimmed = _trim_to_rounds(messages, MAX_ROUNDS)

    try:
        r = _r()
        key = _key(session_id)
        payload = json.dumps(trimmed, ensure_ascii=False)
        r.hset(key, mapping={
            "messages": payload,
            "summary": summary,
            "user_id": str(user_id) if user_id else "",
        })
        r.expire(key, REDIS_SESSION_TTL)
        _touch_index(session_id)
        logger.info(f"会话 {session_id[:8]} 已保存（{len(trimmed)}条消息）")
        if len(trimmed) < len(messages):
            logger.info(f"会话裁剪: {len(messages)}→{len(trimmed)}条（保留{MAX_ROUNDS}轮）")
    except Exception as e:
        logger.error(f"Redis 写入失败: {e}", exc_info=True)


def list_sessions(limit: int = 50) -> list[dict]:
    """列出最近会话（KEYS 扫描为主，ZSet 排序辅助）"""
    try:
        r = _r()

        # ── 第一步：用 KEYS 找到所有 session hash ──
        all_keys = r.keys(_key("*"))
        logger.error(f"[DIAG] KEYS 扫描到 {len(all_keys)} 个 session key")

        if not all_keys:
            return []

        # ── 第二步：逐个读取 hash 内容 ──
        result = []
        for k in all_keys:
            sid = k.decode().split(":")[-1] if isinstance(k, bytes) else k.split(":")[-1]
            data = r.hgetall(_key(sid))
            if not data:
                continue

            msgs_raw = data.get(b"messages", b"[]")
            msgs = json.loads(msgs_raw)
            if not msgs:
                continue

            # 提取预览文本
            preview = ""
            for m in msgs:
                c = str(m.get("content", "")).strip()
                if not c:
                    continue
                if m.get("role") == "user":
                    preview = c[:40]
                    break
                if m.get("role") == "assistant" and not preview:
                    preview = c[:40]
            if not preview:
                preview = "(空会话)"

            uid_raw = data.get(b"user_id", b"")
            uid = uid_raw.decode("utf-8") if uid_raw else None

            result.append({
                "session_id": sid,
                "preview": preview,
                "msg_count": len(msgs),
                "user_id": uid or None,
            })

        logger.error(f"[DIAG] 最终返回 {len(result)} 个有效会话")

        # ── 第三步：按 session_id 倒序（近似按时间） ──
        result.sort(key=lambda x: x["session_id"], reverse=True)
        return result[:limit]

    except Exception as e:
        logger.error(f"列出会话失败: {e}", exc_info=True)
        return []


def load_session(session_id: str) -> dict | None:
    """加载完整会话数据（供 UI 切换历史）"""
    try:
        r = _r()
        data = r.hgetall(_key(session_id))
        if not data:
            return None
        return {
            "session_id": session_id,
            "messages": json.loads(data.get(b"messages", b"[]")),
            "summary": data.get(b"summary", b"").decode("utf-8"),
            "user_id": data.get(b"user_id", b"").decode("utf-8") or None,
        }
    except Exception as e:
        logger.warning(f"加载会话失败: {e}")
        return None


def delete_session(session_id: str) -> bool:
    """删除会话（Redis Hash + ZSet 索引）"""
    try:
        r = _r()
        r.delete(_key(session_id))
        r.zrem(_SESSIONS_INDEX, session_id)
        logger.info(f"会话 {session_id[:8]} 已删除")
        return True
    except Exception as e:
        logger.warning(f"删除会话失败: {e}")
        return False


# ═══════════════════════════════════════════════════════════
#  内部
# ═══════════════════════════════════════════════════════════

def _trim_to_rounds(messages: list[dict], max_rounds: int) -> list[dict]:
    """只保留最近 max_rounds 轮对话。一轮 = 一条user消息开始。"""
    if not messages:
        return messages

    # 从后往前找到第 max_rounds 个 user 消息的位置，从那里开始截断
    rounds = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            rounds += 1
            if rounds > max_rounds:
                return messages[i + 1:]
    return messages  # 不超限，全部保留
