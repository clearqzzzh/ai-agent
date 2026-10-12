"""FastAPI 中间件 — 自动注入 trace_id 到每个请求"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from memory.tracer import start_trace, finish_trace
from loguru import logger
import time


class TraceMiddleware(BaseHTTPMiddleware):
    """每个 HTTP 请求自动创建 trace_id"""

    async def dispatch(self, request: Request, call_next):
        # 从 header 提取或新建 trace_id
        session_id = request.query_params.get("session_id", "unknown")
        trace = start_trace(session_id)
        logger.info(f"[{trace.trace_id}] → {request.method} {request.url.path}")

        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - t0) * 1000

        # 把 trace 汇总注入 response header
        trace_data = finish_trace()
        response.headers["X-Trace-Id"] = trace.trace_id
        response.headers["X-Trace-Elapsed-Ms"] = f"{elapsed:.0f}"

        logger.info(f"[{trace.trace_id}] ← {response.status_code} {elapsed:.0f}ms")
        return response
