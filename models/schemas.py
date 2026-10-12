"""
Pydantic 模型 — 对标 Spring Boot DTO / Entity
请求体校验 + 响应体序列化。
"""

from typing import List

from pydantic import BaseModel, Field


class Message(BaseModel):
    """单条对话消息"""
    role: str = Field(..., description="消息角色：system / user / assistant")
    content: str = Field(..., min_length=1, description="消息内容")


class ChatRequest(BaseModel):
    """用户聊天请求体"""
    messages: List[Message] = Field(..., min_length=1, description="对话消息列表")
    session_id: str | None = Field(default=None, description="会话 ID（不传则自动生成新会话）")
    user_id: int | None = Field(default=None, description="当前登录用户ID")
    system: str | None = Field(default=None, description="System prompt（可选）")
    max_tokens: int = Field(default=2048, ge=1, le=8192, description="最大生成 token 数")


class TraceSpan(BaseModel):
    """单次操作追踪"""
    name: str = Field(..., description="操作名称")
    type: str = Field(..., description="类型: intent / tool_call / llm_generation / agent")
    elapsed_ms: float = Field(..., description="耗时(毫秒)")
    metadata: dict = Field(default={}, description="附加信息")


class ChatResponse(BaseModel):
    """聊天回复响应体"""
    session_id: str = Field(..., description="会话 ID（客户端应保存以便后续请求传入）")
    reply: str = Field(..., description="AI 回复内容")
    model: str = Field(..., description="使用的模型")
    tool_calls_made: List[str] = Field(default=[], description="本次请求调用过的工具名称列表")
    trace_id: str = Field(default="", description="全链路追踪ID")
    trace: List[TraceSpan] = Field(default=[], description="全链路追踪详情")



