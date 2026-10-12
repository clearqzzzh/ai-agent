"""
FastAPI 服务入口 — 对标 Spring Boot @SpringBootApplication（瘦入口）
==================================================================
分层架构：
  config.py          → 配置（对标 application.yml）
  models/schemas.py  → DTO（对标 @Entity / DTO）
  tools/             → Function Calling 工具注册（对标 @Component）
  services/          → 业务逻辑（对标 @Service）
  memory/            → 会话持久化 + Token 管理（对标 Repository + Util）
  rag.py             → RAG 知识库

启动方式：uvicorn main:app --reload

上下文管理（企业级）：
  - Token 感知裁剪：tiktoken 精确计数，超 6000 token 自动裁剪
  - 分层记忆：最近窗口(2500tok)原文保留 + 旧消息摘要压缩 + RAG 长期记忆
  - 摘要合并：新旧摘要递归融合，不丢失早期对话信息
  - 文件持久化：sessions/*.json，服务重启不丢上下文
"""

import json
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config import DEEPSEEK_MODEL, REDIS_URL
from models.schemas import ChatRequest, ChatResponse, TraceSpan
from execution.chat_service import chat_with_tools
from execution.product_service import get_product_token
from execution.vision_service import analyze_image
from access.middleware import TraceMiddleware
from memory.session_store import (
    list_sessions as redis_list,
    load_session as redis_load,
    delete_session as redis_delete,
)


# ═══════════════════════════════════════════════════════
#  FastAPI 应用
# ═══════════════════════════════════════════════════════
app = FastAPI(
    title="Agent Service",
    description="接收用户消息，调用 DeepSeek API（支持 Function Calling），返回 AI 回复。支持多轮对话记忆。",
    version="0.5.0",
)

# CORS — 允许 H5 页面跨域调用
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 注册全链路追踪中间件
app.add_middleware(TraceMiddleware)


@app.on_event("startup")
async def _startup_login():
    """启动时预加载所有模型，避免首次请求阻塞"""
    token = await get_product_token()
    if token:
        logger.info(f"✅ 商城登录就绪 (token: {token[:8]}...)")
    # 预加载 embedding 模型
    try:
        from cognition.engine import preload_model
        preload_model()
        logger.info("✅ Embedding 模型预加载完成")
    except Exception as e:
        logger.warning(f"Embedding 预加载失败: {e}")
    else:
        logger.warning("⚠️ 商城未登录 — 请检查 .env 中的账号密码是否正确")


# ═══════════════════════════════════════════════════════
#  路由（对标 @RestController）
# ═══════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    接收用户消息，调用 DeepSeek API（支持 Function Calling），返回 AI 回复。

    - **messages**: 本轮消息列表，每条包含 role 和 content
    - **session_id**: 会话 ID（不传自动生成）
    - **system**: 可选的 system prompt
    - **max_tokens**: 最大返回 token 数
    """
    session_id = request.session_id or uuid.uuid4().hex[:12]

    try:
        msg_dicts = [{"role": m.role, "content": m.content} for m in request.messages]
        reply, tools_used, trace_id, trace_spans = await chat_with_tools(
            session_id, msg_dicts, request.system, request.max_tokens,
            user_id=request.user_id
        )
        return ChatResponse(
            session_id=session_id,
            reply=reply,
            model=DEEPSEEK_MODEL,
            tool_calls_made=tools_used,
            trace_id=trace_id,
            trace=[TraceSpan(
                name=s["name"], type=s["type"],
                elapsed_ms=s["elapsed_ms"], metadata=s.get("metadata", {})
            ) for s in trace_spans],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"DeepSeek API 错误: {str(e)}")


@app.post("/vision")
async def vision(request: dict):
    """图片识别 — 直调千问 VL，不走 Agent。返回识别文本。"""
    image_url = request.get("image_url", "")
    question = request.get("question", "")
    if not image_url:
        raise HTTPException(status_code=400, detail="缺少 image_url")
    analysis = await analyze_image(image_url, question)
    return {"analysis": analysis}


@app.get("/debug/state/{session_id}")
async def debug_state(session_id: str):
    """Debug: 查看 session 的 user_id 状态"""
    from memory.session_store import get_or_create_session
    data = get_or_create_session(session_id)
    return {"user_id": data.get("user_id"), "username": data.get("username",""), "msg_count": len(data.get("messages",[]))}


@app.get("/debug/sessions")
async def debug_sessions():
    """诊断端点：直接查看 Redis 中的会话原始数据"""
    from memory.session_store import _r, _key, _SESSIONS_INDEX
    try:
        r = _r()
        # 1. ZSet 成员
        zset_members = r.zrevrange(_SESSIONS_INDEX, 0, 9)
        zset_count = r.zcard(_SESSIONS_INDEX)
        # 2. 所有 agent:session:* 键
        all_keys = r.keys("agent:session:*")
        # 3. 检查每个 ZSet 成员对应的 hash 是否存在
        zset_detail = []
        for sid_bytes in zset_members:
            sid = sid_bytes.decode() if isinstance(sid_bytes, bytes) else str(sid_bytes)
            exists = r.exists(_key(sid))
            zset_detail.append({"session_id": sid, "hash_exists": bool(exists)})
        # 4. 对存在的 hash 读取内容摘要
        hash_detail = []
        for k in all_keys[:5]:
            key_str = k.decode() if isinstance(k, bytes) else str(k)
            data = r.hgetall(key_str)
            msgs_raw = data.get(b"messages", b"[]")
            msgs = json.loads(msgs_raw) if isinstance(msgs_raw, (bytes, str)) else []
            hash_detail.append({
                "key": key_str,
                "msg_count": len(msgs),
                "has_user_id": bool(data.get(b"user_id", b"")),
            })
        return {
            "zset_total": zset_count,
            "zset_top10": zset_detail,
            "hash_keys_count": len(all_keys),
            "hash_details": hash_detail,
            "redis_url": REDIS_URL,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/sessions")
async def list_sessions():
    """列出历史会话（Redis ZSet，按时间倒序，最近50条）"""
    return {"sessions": redis_list(50)}


@app.get("/sessions/{session_id}")
async def load_session(session_id: str):
    """加载指定会话（Redis）"""
    data = redis_load(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="会话不存在")
    return data


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    """删除指定会话"""
    ok = redis_delete(session_id)
    if not ok:
        raise HTTPException(status_code=500, detail="删除失败")
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("access.main:app", host="0.0.0.0", port=8001, reload=True)
