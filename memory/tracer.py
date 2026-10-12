"""
全链路追踪 — Trace 上下文管理
trace_id 贯穿 supervisor → agent → tool_call → LLM generation
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """一次操作的时间跨度"""
    name: str           # 操作名称 (supervisor_decision / product_agent / tool.search_product / llm_generate)
    type: str           # 类型: intent / tool_call / llm_generation / agent
    start_time: float
    end_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def finish(self) -> "Span":
        self.end_time = time.perf_counter()
        return self


@dataclass
class Trace:
    """一次完整请求的追踪"""
    trace_id: str
    session_id: str
    spans: list[Span] = field(default_factory=list)
    start_time: float = field(default_factory=time.perf_counter)

    def add_span(self, name: str, span_type: str, metadata: dict[str, Any] | None = None) -> Span:
        """创建并添加一个新span，返回span对象（调用方在操作完成后调 span.finish()）"""
        span = Span(
            name=name,
            type=span_type,
            start_time=time.perf_counter(),
            metadata=metadata or {},
        )
        self.spans.append(span)
        return span

    def to_dict(self) -> dict:
        """转为前端可展示的格式"""
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "total_elapsed_ms": (time.perf_counter() - self.start_time) * 1000,
            "spans": [
                {
                    "name": s.name,
                    "type": s.type,
                    "elapsed_ms": round(s.elapsed_ms, 1),
                    "metadata": s.metadata,
                }
                for s in self.spans
            ],
        }


# ══════════════════════════════════════════════════════════════
#  线程本地存储 — 当前请求的 trace（避免跨请求污染）
# ══════════════════════════════════════════════════════════════

import threading

_trace_local = threading.local()


def start_trace(session_id: str) -> Trace:
    """开始一次追踪"""
    trace = Trace(
        trace_id=uuid.uuid4().hex[:12],
        session_id=session_id,
    )
    _trace_local.current = trace
    return trace


def get_current_trace() -> Trace | None:
    """获取当前线程的trace"""
    return getattr(_trace_local, "current", None)


def finish_trace() -> dict:
    """结束追踪，返回汇总dict"""
    trace = get_current_trace()
    if not trace:
        return {}
    return trace.to_dict()
