"""
LangGraph Supervisor 多Agent编排图
===================================
Supervisor 模式：1个调度Agent + 6个执行Agent

节点：
  __start__ → supervisor → product_consult → supervisor → ... → __end__
                          → order_ops      → supervisor → ...
                          → aftersale      → supervisor → ...
                          → pet_health     → supervisor → ...
                          → knowledge_retrieval → supervisor → ...
                          → unknown_intent → supervisor → ...

状态：
  AgentState(messages, next_agent, final_reply, tool_calls_made, trace_id, iteration)
"""

import json
import re
import time
from typing import Literal

from langgraph.graph import StateGraph, END
from langsmith import traceable
from loguru import logger
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from orchestration.state import AgentState
from orchestration.prompts import (
    SUPERVISOR_PROMPT,
    build_product_consult_agent_prompt,
    build_order_ops_agent_prompt,
    build_aftersale_agent_prompt,
    build_pet_health_agent_prompt,
    build_knowledge_retrieval_agent_prompt,
    build_unknown_intent_agent_prompt,
)
from orchestration.tool_sets import (
    PRODUCT_CONSULT_TOOLS,
    ORDER_OPS_TOOLS,
    AFTERSALE_TOOLS,
    PET_HEALTH_TOOLS,
    KNOWLEDGE_RETRIEVAL_TOOLS,
    UNKNOWN_INTENT_TOOLS,
)
from execution.dispatcher import dispatch_tool

_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=60.0, max_retries=2)
MAX_ITERATIONS = 8


# ══════════════════════════════════════════════════════════════
#  Supervisor 节点 — 意图识别 + 路由决策
# ══════════════════════════════════════════════════════════════

@traceable(run_type="chain", name="supervisor_intent", tags=["supervisor", "intent-routing"])
def _supervisor_node(state: AgentState) -> dict:
    """分析用户最新消息，决定下一个Agent或结束"""
    t0 = time.perf_counter()
    trace_id = state.get("trace_id", "")

    # 超过最大迭代次数 → 强制结束 + 兜底回复
    iteration = state.get("iteration", 0) + 1
    if iteration > MAX_ITERATIONS:
        logger.warning(f"[{trace_id}] 超过最大迭代次数 {MAX_ITERATIONS}，强制结束")
        return {
            "next_agent": "FINISH",
            "iteration": iteration,
            "final_reply": "抱歉，这个问题有点复杂，我暂时没能帮您解决。建议您联系人工客服获取更专业的帮助，或者换个方式描述一下您的问题，我再试试帮您处理。🐾",
        }

    # 之前的Agent已经给了最终回复 → 结束
    if state.get("final_reply"):
        return {"iteration": iteration}

    # 取最新几条消息给 supervisor 做意图判断
    recent = state["messages"][-6:]  # 只取最近6条
    decide_messages = [{"role": "system", "content": SUPERVISOR_PROMPT}]
    for m in recent:
        if m.get("role") in ("user", "assistant"):
            decide_messages.append({
                "role": m["role"],
                "content": str(m.get("content", ""))[:300]  # 截断长消息
            })
    decide_messages.append({"role": "user", "content": "请判断应该路由到哪个Agent（product_consult / order_ops / aftersale / pet_health / knowledge_retrieval / unknown_intent / FINISH）"})

    # 硬规则：包含退货/退款/售后关键词 → 强制 aftersale
    last_user = ""
    for m in reversed(state["messages"]):
        if m.get("role") == "user":
            last_user = str(m.get("content", ""))
            break
    if any(w in last_user for w in ["退货", "退款", "退钱", "不想要了", "售后", "退掉", "取消订单", "退货进度"]):
        logger.info(f"[{trace_id}] Supervisor 硬规则: 售后关键词 → aftersale")
        return {"next_agent": "aftersale", "iteration": iteration}

    # 硬规则：包含订单/购物车/地址/物流关键词 → 强制 order_ops
    if any(w in last_user for w in ["订单", "下单", "购物车", "地址", "我的订单", "结算", "买", "物流", "快递", "包裹", "发货", "付款", "支付", "确认收货", "注册"]):
        logger.info(f"[{trace_id}] Supervisor 硬规则: 订单关键词 → order_ops")
        return {"next_agent": "order_ops", "iteration": iteration}

    # 硬规则：包含症状/医疗关键词 → 强制 pet_health
    if any(w in last_user for w in ["呕吐", "拉稀", "腹泻", "生病", "不舒服", "医院",
                                      "掉毛", "流泪", "咳嗽", "打喷嚏", "跛行", "不吃",
                                      "精神差", "皮肤红", "得了", "就医", "中毒"]):
        logger.info(f"[{trace_id}] Supervisor 硬规则: 医疗关键词 → pet_health")
        return {"next_agent": "pet_health", "iteration": iteration}

    # 硬规则：包含喂养知识关键词 → 强制 knowledge_retrieval
    if any(w in last_user for w in ["能不能喂", "怎么养", "吃什么好", "营养", "喂养", "护理知识"]):
        logger.info(f"[{trace_id}] Supervisor 硬规则: 知识关键词 → knowledge_retrieval")
        return {"next_agent": "knowledge_retrieval", "iteration": iteration}

    try:
        resp = _client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=decide_messages,
            max_tokens=20,
            timeout=15.0,
        )
        decision = resp.choices[0].message.content.strip().lower()
    except Exception as e:
        logger.warning(f"[{trace_id}] Supervisor LLM调用失败: {e}，默认结束")
        decision = "finish"

    elapsed = (time.perf_counter() - t0) * 1000

    # 解析决策
    if "product_consult" in decision:
        next_agent = "product_consult"
    elif "order_ops" in decision:
        next_agent = "order_ops"
    elif "aftersale" in decision:
        next_agent = "aftersale"
    elif "pet_health" in decision:
        next_agent = "pet_health"
    elif "knowledge_retrieval" in decision:
        next_agent = "knowledge_retrieval"
    elif "unknown_intent" in decision:
        next_agent = "unknown_intent"
    else:
        next_agent = "FINISH"

    logger.info(
        f"[{trace_id}] Supervisor(#{iteration}): {decision!r} → {next_agent}, "
        f"耗时{elapsed:.0f}ms"
    )
    return {"next_agent": next_agent, "iteration": iteration}


# ══════════════════════════════════════════════════════════════
#  各Agent节点
# ══════════════════════════════════════════════════════════════

async def _product_consult_node(state: AgentState) -> dict:
    """商品咨询专员：用 search_product / analyze_image / get_reviews / add_review 回复用户"""
    return await _agent_node(
        state, PRODUCT_CONSULT_TOOLS, "product_consult",
        prompt_builder=build_product_consult_agent_prompt,
    )


async def _order_ops_node(state: AgentState) -> dict:
    """下单操作专员：用地址/下单/购物车/订单/物流/支付工具回复用户"""
    return await _agent_node(
        state, ORDER_OPS_TOOLS, "order_ops",
        prompt_builder=build_order_ops_agent_prompt,
    )


async def _aftersale_node(state: AgentState) -> dict:
    """售后处理专员：用退货退款工具回复用户"""
    return await _agent_node(
        state, AFTERSALE_TOOLS, "aftersale",
        prompt_builder=build_aftersale_agent_prompt,
    )


async def _pet_health_node(state: AgentState) -> dict:
    """宠物健康专员：用医疗知识库 + 商品搜索回复用户"""
    return await _agent_node(
        state, PET_HEALTH_TOOLS, "pet_health",
        prompt_builder=build_pet_health_agent_prompt,
    )


async def _knowledge_retrieval_node(state: AgentState) -> dict:
    """知识检索专员：用喂养知识库回复用户"""
    return await _agent_node(
        state, KNOWLEDGE_RETRIEVAL_TOOLS, "knowledge_retrieval",
        prompt_builder=build_knowledge_retrieval_agent_prompt,
    )


async def _unknown_intent_node(state: AgentState) -> dict:
    """未知意图兜底：无工具，纯文本回复"""
    return await _agent_node(
        state, UNKNOWN_INTENT_TOOLS, "unknown_intent",
        prompt_builder=build_unknown_intent_agent_prompt,
    )


# ══════════════════════════════════════════════════════════════
#  通用 Agent 节点逻辑
# ══════════════════════════════════════════════════════════════

def _fix_tool_messages(msgs: list[dict]) -> list[dict]:
    """清理所有 tool_calls / tool 历史消息，避免 DeepSeek 400 错误。
    当前轮次的工具调用由 agent 节点重新添加，历史中的工具消息无意义。"""
    result = []
    for m in msgs:
        role = m.get("role", "")
        # 跳过所有 tool 角色消息（tool result）
        if role == "tool":
            continue
        # 跳过带 tool_calls 的 assistant 消息（历史工具调用已过期）
        if role == "assistant" and m.get("tool_calls"):
            continue
        # assistant 消息 content 为 null → 空字符串
        if role == "assistant" and m.get("content") is None:
            m["content"] = ""
        result.append(m)
    return result


# ══════════════════════════════════════════════════════════════
#  DSML 文本工具调用解析器（Fallback）
#  当模型输出 DSML 格式而非结构化 tool_calls 时使用
# ══════════════════════════════════════════════════════════════

def _parse_dsml_calls(content: str) -> list[dict] | None:
    """
    解析 DSML 格式的文本工具调用，返回与 OpenAI tool_calls 结构兼容的列表。
    返回 None 表示未检测到 DSML 格式。

    DSML 格式示例：
        <|DSML|calls>
        <|DSML|invoke name="request_refund">
        <|DSML|parameter name="order_id" string="false">10</|DSML|parameter>
        <|DSML|parameter name="reason" string="true">不想要了</|DSML|parameter>
        </|DSML|invoke>
        </|DSML|calls>
    """
    if not content or '<|DSML|calls>' not in content:
        return None

    calls = []
    # 提取所有 invoke 块
    invoke_pattern = re.compile(
        r'<\|DSML\|invoke\s+name="([^"]+)"\s*>(.*?)</\|DSML\|invoke>',
        re.DOTALL
    )
    param_pattern = re.compile(
        r'<\|DSML\|parameter\s+name="([^"]+)"\s+string="([^"]*)"\s*>(.*?)</\|DSML\|parameter>',
        re.DOTALL
    )

    for inv_match in invoke_pattern.finditer(content):
        tool_name = inv_match.group(1)
        invoke_body = inv_match.group(2)
        args = {}
        for pm in param_pattern.finditer(invoke_body):
            pname = pm.group(1)
            is_string = pm.group(2).lower() == 'true'
            pvalue = pm.group(3).strip()
            if is_string:
                args[pname] = pvalue
            else:
                # 尝试转为数字
                try:
                    args[pname] = int(pvalue)
                except ValueError:
                    try:
                        args[pname] = float(pvalue)
                    except ValueError:
                        args[pname] = pvalue

        calls.append({
            "id": f"dsml_{tool_name}_{len(calls)}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })

    return calls if calls else None


@traceable(run_type="chain", name="agent_execution", tags=["agent", "tool-calling"])
async def _agent_node(
    state: AgentState,
    tools: list[dict],
    agent_name: str,
    prompt_builder,
) -> dict:
    """通用Agent节点：送消息+工具给LLM，拿到回复或工具调用结果后返回"""
    trace_id = state.get("trace_id", "")
    t0 = time.perf_counter()
    tool_calls_made: list[str] = []

    # 拼装消息（包含历史 + agent专用prompt）
    messages = state.get("messages", []).copy()

    # 从 session 读 user_id
    sid = state.get("session_id", "")
    uid = None
    if sid:
        from memory.session_store import get_or_create_session
        sess = get_or_create_session(sid)
        uid = sess.get("user_id")

    # 已登录 → 隐藏 lookup_user
    active_tools = tools
    if uid and uid > 0:
        active_tools = [t for t in tools if t["function"]["name"] != "lookup_user"]

    uname = state.get("username", "") or ""
    system_content = prompt_builder(user_id=uid, username=uname)
    full_messages = [{"role": "system", "content": system_content}]
    full_messages.extend(messages)

    # 修复：DeepSeek 要求 tool_calls 和 tool result 成对出现
    # 裁剪可能导致孤立的 tool_calls → 抛 400。清理掉不成对的。
    full_messages = _fix_tool_messages(full_messages)

    # LLM 调用（支持Function Calling）
    try:
        llm_t0 = time.perf_counter()
        response = _client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=full_messages,
            max_tokens=1024,
            tools=active_tools,
            timeout=45.0,
        )
        llm_elapsed = (time.perf_counter() - llm_t0) * 1000
        choice = response.choices[0]
        msg = choice.message
    except Exception as e:
        err = str(e)
        logger.error(f"[{trace_id}] {agent_name} LLM调用失败: {err}")
        if '429' in err or 'rate' in err.lower():
            hint = 'AI 服务繁忙，请稍后再试 😅'
        elif 'timeout' in err.lower() or 'timed out' in err.lower():
            hint = 'AI 思考超时，请换个简单的问题试试 ⏱️'
        elif '401' in err or '403' in err or 'auth' in err.lower():
            hint = 'AI 服务配置异常，请联系管理员 🔧'
        else:
            hint = f'AI 服务暂时不可用（{err[:60]}），请稍后重试'
        return {"final_reply": hint}

    # 没有结构化工具调用 → 尝试 DSML 文本解析 fallback
    dsml_calls = None
    if not msg.tool_calls and msg.content:
        dsml_calls = _parse_dsml_calls(msg.content)

    if not msg.tool_calls and not dsml_calls:
        # 纯文本回复
        reply_content = msg.content or ""
        assistant_msg = {"role": "assistant", "content": reply_content}
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[{trace_id}] {agent_name}: 文本回复({len(reply_content)}字), "
            f"LLM耗时{llm_elapsed:.0f}ms, 总耗时{elapsed:.0f}ms"
        )
        return {
            "messages": [assistant_msg],
            "final_reply": reply_content,
            "tool_calls_made": tool_calls_made,
        }

    # 有工具调用（结构化 或 DSML 解析）→ 执行工具 → 再调LLM生成回复
    if dsml_calls and not msg.tool_calls:
        # DSML fallback：用解析出的工具调用替代
        logger.info(f"[{trace_id}] {agent_name}: 检测到DSML格式，解析出{len(dsml_calls)}个工具调用")
        # 清理 content 中的 DSML 标记，只保留纯文本部分（如果有）
        clean_content = re.sub(r'<\|DSML\|.*?>', '', msg.content or '').strip()
        assistant_msg = {
            "role": "assistant",
            "content": clean_content if clean_content else None,
            "tool_calls": dsml_calls,
        }
        # 构造兼容的 tool_calls 对象（用 SimpleNamespace 模拟）
        from types import SimpleNamespace
        synthetic_calls = [
            SimpleNamespace(
                id=tc["id"],
                function=SimpleNamespace(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in dsml_calls
        ]
    else:
        assistant_msg = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        }
        synthetic_calls = msg.tool_calls

    full_messages.append(assistant_msg)

    for tc in synthetic_calls:
        tool_t0 = time.perf_counter()
        tool_name = tc.function.name
        tool_args = json.loads(tc.function.arguments)

        if uid and uid > 0 and tool_name in ("get_user_address", "create_order", "save_user_address", "generate_order_card", "get_orders", "get_order_detail", "get_cart", "request_refund", "check_refund_eligibility", "get_refund_info"):
            tool_args["user_id"] = uid

        try:
            result = await dispatch_tool(tool_name, tool_args)
        except Exception as e:
            result = {"success": False, "error": str(e)}

        tool_elapsed = (time.perf_counter() - tool_t0) * 1000
        tool_calls_made.append(tool_name)
        full_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result, ensure_ascii=False),
        })
        logger.info(
            f"[{trace_id}] {agent_name}.{tool_name}(...): "
            f"耗时{tool_elapsed:.0f}ms"
        )

    # 工具执行完 → 立即调LLM生成最终回复（不带tools，强制文本输出）
    llm2_t0 = time.perf_counter()
    try:
        resp2 = _client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=full_messages,
            max_tokens=1024,
            timeout=45.0,
        )
        reply_content = resp2.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"[{trace_id}] {agent_name} 生成回复失败: {e}")
        reply_content = f"已为您完成{len(tool_calls_made)}项操作，请查看结果。"

    llm2_elapsed = (time.perf_counter() - llm2_t0) * 1000
    full_messages.append({"role": "assistant", "content": reply_content})
    elapsed = (time.perf_counter() - t0) * 1000

    logger.info(
        f"[{trace_id}] {agent_name}: {len(synthetic_calls)}工具+生成回复({len(reply_content)}字), "
        f"LLM决策{llm_elapsed:.0f}ms + 生成{llm2_elapsed:.0f}ms, 总{elapsed:.0f}ms"
    )
    return {
        "messages": [assistant_msg]
        + [m for m in full_messages if m.get("role") == "tool"
           and m.get("tool_call_id") == assistant_msg["tool_calls"][0]["id"]]
        + [{"role": "assistant", "content": reply_content}],
        "final_reply": reply_content,
        "tool_calls_made": tool_calls_made,
    }


# ══════════════════════════════════════════════════════════════
#  路由函数
# ══════════════════════════════════════════════════════════════

def _route_decision(state: AgentState) -> Literal["product_consult", "order_ops", "aftersale", "pet_health", "knowledge_retrieval", "unknown_intent", "__end__"]:
    """根据 supervisor 的 next_agent 决定流向"""
    agent = state.get("next_agent", "FINISH")
    if agent == "product_consult":
        return "product_consult"
    elif agent == "order_ops":
        return "order_ops"
    elif agent == "aftersale":
        return "aftersale"
    elif agent == "pet_health":
        return "pet_health"
    elif agent == "knowledge_retrieval":
        return "knowledge_retrieval"
    elif agent == "unknown_intent":
        return "unknown_intent"
    return "__end__"


def _after_agent_route(state: AgentState) -> Literal["supervisor", "__end__"]:
    """Agent执行完后：如果有 final_reply → 结束，否则 → 回到 supervisor 继续"""
    if state.get("final_reply"):
        return "__end__"
    return "supervisor"


# ══════════════════════════════════════════════════════════════
#  构建图 — 对外暴露的唯一入口
# ══════════════════════════════════════════════════════════════

def create_supervisor_graph() -> StateGraph:
    """构建Supervisor多Agent图（7-Agent架构）"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("supervisor", _supervisor_node)
    workflow.add_node("product_consult", _product_consult_node)
    workflow.add_node("order_ops", _order_ops_node)
    workflow.add_node("aftersale", _aftersale_node)
    workflow.add_node("pet_health", _pet_health_node)
    workflow.add_node("knowledge_retrieval", _knowledge_retrieval_node)
    workflow.add_node("unknown_intent", _unknown_intent_node)

    # 入口 → supervisor
    workflow.set_entry_point("supervisor")

    # supervisor → 6个Agent / END
    workflow.add_conditional_edges(
        "supervisor",
        _route_decision,
        {
            "product_consult": "product_consult",
            "order_ops": "order_ops",
            "aftersale": "aftersale",
            "pet_health": "pet_health",
            "knowledge_retrieval": "knowledge_retrieval",
            "unknown_intent": "unknown_intent",
            "__end__": END,
        },
    )

    # 每个Agent执行完后 → supervisor（继续）或 END
    for agent_name in ["product_consult", "order_ops", "aftersale", "pet_health", "knowledge_retrieval", "unknown_intent"]:
        workflow.add_conditional_edges(
            agent_name,
            _after_agent_route,
            {"supervisor": "supervisor", "__end__": END},
        )

    return workflow.compile()
