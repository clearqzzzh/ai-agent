"""LangGraph AgentState — 多Agent共享状态"""
from typing import TypedDict, Annotated
from operator import add


class AgentState(TypedDict):
    messages: Annotated[list[dict], add]
    next_agent: str
    tool_results: list[dict]
    final_reply: str
    tool_calls_made: Annotated[list[str], add]
    trace_id: str
    iteration: int
    user_id: int       # 当前登录用户ID
    session_id: str     # 会话ID
    username: str         # 用户名
