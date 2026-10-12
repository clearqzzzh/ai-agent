"""
对话服务 — LangGraph Supervisor 多Agent编排
==========================================
取代旧版单Agent工具调用循环，改为 Supervisor → ProductAgent / OrderAgent 图。
"""

from loguru import logger
from langsmith import traceable

from config import DEEPSEEK_MODEL
from memory.session_store import get_or_create_session, save_session
from memory.token_utils import token_based_trim
from memory.tracer import start_trace, finish_trace


@traceable(run_type="chain", name="chat_with_tools", tags=["supervisor", "main-entry"])
async def chat_with_tools(
    session_id: str,
    incoming: list[dict],
    system: str | None,
    max_tokens: int,
    user_id: int | None = None,
) -> tuple[str, list[str], str, list[dict]]:
    """
    LangGraph Supervisor 多Agent编排入口。
    返回 (最终回复, 工具调用列表, trace_id, trace详情)。

    流程：
      1. 加载会话 + Token裁剪
      2. 启动 trace
      3. 构建 graph 初始状态
      4. graph.invoke() → 多Agent协作
      5. 保存会话 + 返回结果
    """
    # 1. 加载会话
    data = get_or_create_session(session_id)
    history: list[dict] = data["messages"]
    old_summary: str = data.get("summary", "")

    if user_id is not None:
        data["user_id"] = user_id
    current_user_id: int | None = data.get("user_id")
    username = data.get("username", "")

    # 有 user_id 但没用户名 → 查用户信息
    if current_user_id and not username:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"http://localhost:8083/user/{current_user_id}")
                if r.status_code == 200 and r.json().get("success"):
                    u = r.json()["data"]
                    username = u.get("username", "")
                    data["username"] = username
                    logger.info(f"[用户] 查询成功: id={current_user_id}, name={username}")
                else:
                    logger.warning(f"[用户] 查询失败: status={r.status_code}")
        except Exception as e:
            logger.warning(f"[用户] 查询异常: {e}")

    # 已登录用户：查订单/地址等主动注入上下文
    if current_user_id and incoming and incoming[-1].get("content"):
        last_msg = incoming[-1]["content"]
        if any(w in last_msg for w in ["我的订单", "订单", "查订单"]):
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5) as c2:
                    r = await c2.get(f"http://localhost:8084/api/order", params={"userId": current_user_id})
                    if r.status_code == 200:
                        od = r.json()
                        orders = od.get("data", []) if od.get("success") else []
                        if orders:
                            summary = f"[系统] {username}的订单：共{len(orders)}条。" + "、".join(
                                f"#{o.get('id','')} ¥{o.get('totalAmount',0)} {o.get('statusText','')}"
                                for o in orders[:5]
                            )
                            history.append({"role": "system", "content": summary})
            except Exception:
                pass

    # 有用户信息时注入
    if current_user_id and username:
        history.append({
            "role": "system",
            "content": f"[当前用户] {username}(ID:{current_user_id})已登录，禁止问手机号。"
        })

    # 2. 拼接 + Token 裁剪
    history.extend(incoming)
    history, new_summary = token_based_trim(
        history, old_summary,
        _get_client()
    )

    # 3. 启动链路追踪
    trace = start_trace(session_id)
    trace_id = trace.trace_id

    # 4. 构建 graph 初始状态
    from orchestration.supervisor_graph import create_supervisor_graph
    graph = create_supervisor_graph()

    logger.info(f"[会话 DEBUG] param_user_id={user_id}, current_user_id={current_user_id}, data_keys={list(data.keys())}, data_user_id={data.get('user_id')}")
    initial_state = {
        "messages": history,
        "next_agent": "",
        "tool_results": [],
        "final_reply": "",
        "tool_calls_made": [],
        "trace_id": trace_id,
        "iteration": 0,
        "user_id": current_user_id or 0,
        "session_id": session_id,
        "username": username or "",
    }

    # 5. 执行多Agent图（ainvoke = async invoke）
    try:
        span = trace.add_span("langgraph_supervisor", "agent")
        result = await graph.ainvoke(initial_state, {"recursion_limit": 15})
        span.finish()
    except Exception as e:
        logger.error(f"[{trace_id}] graph执行失败: {e}")
        return f"抱歉，服务暂时不可用：{str(e)}", [], trace_id, []

    # 6. 提取结果
    final_reply = result.get("final_reply", "")
    tool_calls_made = result.get("tool_calls_made", [])
    result_messages = result.get("messages", [])

    # 如果 graph 没有生成 final_reply，取最后一条 assistant 消息
    if not final_reply:
        for m in reversed(result_messages):
            if m.get("role") == "assistant" and m.get("content"):
                final_reply = m["content"]
                break

    if not final_reply:
        final_reply = "好的，请问还有什么可以帮您的？"

    # 7. 保存会话（graph中间消息 + 最终回复 全量写入，过滤 tool 消息）
    history.extend(result_messages)
    if not any(m.get("role") == "assistant" and m.get("content") == final_reply
               for m in history[-5:]):
        history.append({"role": "assistant", "content": final_reply})
    # 过滤掉 tool 角色的消息（工具调用原始结果，前端不需要显示）
    history = [m for m in history if m.get("role") != "tool"]
    save_session(session_id, history, new_summary, current_user_id)

    # 8. 生成 trace
    trace_data = finish_trace()

    logger.info(
        f"[{trace_id}] 完成: {len(tool_calls_made)}次工具调用, "
        f"回复{len(final_reply)}字, 总耗时{trace_data.get('total_elapsed_ms', 0):.0f}ms"
    )

    return final_reply, tool_calls_made, trace_id, trace_data.get("spans", [])


# ═══════════════════════════════════════════════════════════
#  内部工具
# ═══════════════════════════════════════════════════════════

_client = None

def _get_client():
    """懒加载 OpenAI 客户端"""
    global _client
    if _client is None:
        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
        from openai import OpenAI
        _client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=30.0, max_retries=1)
    return _client
