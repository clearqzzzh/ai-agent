"""
Token 工具 — 对标 Spring Boot @Utility
tiktoken 精确计数 + 分层记忆裁剪 + 摘要合并。
"""

import json

import tiktoken
from loguru import logger

from config import MAX_HISTORY_TOKENS, RECENT_WINDOW_TOKENS, SUMMARY_MAX_TOKENS, DEEPSEEK_MODEL

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """精确计算文本 token 数。"""
    return len(_encoder.encode(text))


def msg_tokens(msg: dict) -> int:
    """计算单条消息的 token 数。"""
    return count_tokens(json.dumps(msg, ensure_ascii=False))


def messages_tokens(messages: list[dict]) -> int:
    """计算消息列表总 token 数。"""
    return sum(msg_tokens(m) for m in messages)


def _summarize_history(messages: list[dict], existing_summary: str, client) -> str:
    """调用 LLM 做摘要压缩；已有摘要则融合合并。"""
    if not messages:
        return existing_summary

    if existing_summary:
        system_prompt = (
            f"已有的对话要点：\n{existing_summary}\n\n"
            f"以下是新的对话内容，请将新内容融合进已有要点中，输出一个新的总结（不超过300字）：\n"
        )
    else:
        system_prompt = (
            "请总结以下对话的要点（不超过200字），包括：用户问了什么、AI 做了什么操作、得出的结论。"
            "只输出总结内容，不要加任何前缀或说明。"
        )

    prompt = "\n\n".join(
        f"[{m['role']}]: {str(m.get('content', '')[:500])}"
        for m in messages if m.get("content")
    )
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=SUMMARY_MAX_TOKENS,
        )
        return response.choices[0].message.content or existing_summary
    except Exception:
        return existing_summary or f"（之前有 {sum(1 for m in messages if m['role']=='user')} 轮对话）"


def token_based_trim(history: list[dict], existing_summary: str, client) -> tuple[list[dict], str]:
    """
    分层记忆裁剪：
      1. 总 token ≤ MAX_HISTORY_TOKENS → 不处理
      2. 超出 → 从后往前保留 RECENT_WINDOW_TOKENS 的原始消息
      3. 更早的消息 → 合并进摘要
    """
    total = messages_tokens(history)
    if total <= MAX_HISTORY_TOKENS:
        return history, existing_summary

    recent: list[dict] = []
    recent_tokens = 0
    old: list[dict] = []

    for msg in reversed(history):
        t = msg_tokens(msg)
        if recent_tokens + t <= RECENT_WINDOW_TOKENS:
            recent.insert(0, msg)
            recent_tokens += t
        else:
            old.insert(0, msg)

    new_summary = _summarize_history(old, existing_summary, client) if old else existing_summary

    result: list[dict] = []
    if new_summary:
        result.append({"role": "user", "content": f"[对话历史摘要]：{new_summary}"})
    result.extend(recent)

    logger.info(
        f"[上下文管理] 总token={total} > {MAX_HISTORY_TOKENS}, "
        f"保留最近{len(recent)}条({recent_tokens}tok), "
        f"压缩旧{len(old)}条 → 摘要{count_tokens(new_summary)}tok"
    )
    return result, new_summary
