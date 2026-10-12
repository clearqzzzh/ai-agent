"""
工具调度器 — 对标 Spring Boot Service 层的 @Service
根据工具名路由到具体实现。
支持：自动重试（指数退避）+ 降级策略
"""

import asyncio
import time

import httpx
from loguru import logger

from config import ORDER_SERVICE_URL, PRODUCT_API_URL
from cognition.engine import search_knowledge, search_medical_knowledge
from execution.product_service import search_product_api

# ── 服务地址 ──────────────────────────────────────────────
# 从 PRODUCT_API_URL 推导基础地址: http://localhost:8082/api/product/search → http://localhost:8082
_PRODUCT_BASE = PRODUCT_API_URL.rsplit("/api/", 1)[0]
PAY_SERVICE_URL = _PRODUCT_BASE.replace(":8082", ":8085")  # pay-service 端口 8085
USER_SERVICE_URL = _PRODUCT_BASE.replace(":8082", ":8083")  # user-service 端口 8083

# ── 重试配置 ──────────────────────────────────────────────
MAX_RETRIES = 2                # 最大重试次数
RETRY_BASE_DELAY = 0.5         # 首次重试延迟（秒），指数退避
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    ConnectionRefusedError,
)

# ── 降级映射 ──────────────────────────────────────────────
# 工具失败后尝试的降级方案（返回友好提示 + 降级结果）
_DEGRADATION_MAP = {
    "search_product": "_degrade_search_product",
    "get_orders": "_degrade_get_orders",
    "get_order_detail": "_degrade_get_order_detail",
    "get_cart": "_degrade_get_cart",
    "track_logistics": "_degrade_track_logistics",
}

# 写操作不重试（避免重复提交）
_NO_RETRY_TOOLS = {"save_user_address", "create_order", "update_cart", "remove_cart",
                   "pay_order", "request_refund", "add_review", "register_user", "confirm_receive"}


async def _retry_async(fn, tool_name: str, max_retries: int = MAX_RETRIES):
    """
    带指数退避的异步重试包装器。
    仅对网络异常重试，业务异常（如参数错误）不重试。
    """
    last_error = None
    for attempt in range(1 + max_retries):
        try:
            return await fn()
        except RETRYABLE_EXCEPTIONS as e:
            last_error = e
            if tool_name in _NO_RETRY_TOOLS or attempt >= max_retries:
                break
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                f"[重试] {tool_name} 第{attempt+1}次失败({type(e).__name__})，"
                f"{delay:.1f}s 后重试..."
            )
            await asyncio.sleep(delay)
        except Exception as e:
            # 非网络异常 → 不重试，直接抛出
            raise

    # 重试耗尽
    logger.error(f"[重试] {tool_name} 重试{max_retries}次后仍失败: {last_error}")
    raise last_error


async def dispatch_tool(tool_name: str, tool_args: dict) -> dict:
    """
    根据工具名称分发到对应的处理函数。
    自动重试网络异常 + 失败后降级。
    """
    handler = _get_handler(tool_name)
    if handler is None:
        return {"error": f"未知工具: {tool_name}"}

    try:
        # 带重试的执行
        result = await _retry_async(
            lambda: handler(tool_args),
            tool_name,
        )
        return result

    except Exception as e:
        logger.warning(f"[调度] {tool_name} 最终失败: {e}")

        # 尝试降级
        degrade_fn_name = _DEGRADATION_MAP.get(tool_name)
        if degrade_fn_name:
            degrade_fn = globals().get(degrade_fn_name)
            if degrade_fn:
                logger.info(f"[降级] {tool_name} → {degrade_fn_name}")
                try:
                    degrade_result = await degrade_fn(tool_args, str(e))
                    return degrade_result
                except Exception as de:
                    logger.error(f"[降级] {degrade_fn_name} 也失败: {de}")

        return {"success": False, "error": f"{tool_name} 调用失败: {e}"}


def _get_handler(tool_name: str):
    """根据工具名返回对应的处理函数。"""
    handlers = {
        "search_product": search_product_api,
        "search_knowledge": lambda args: search_knowledge(args.get("query", "")),
        "search_medical_knowledge": lambda args: search_medical_knowledge(args.get("query", "")),
        "lookup_user": _lookup_user,
        "get_user_address": _get_user_address,
        "save_user_address": _save_user_address,
        "create_order": _create_order,
        "analyze_image": _analyze_image,
        "get_orders": _get_orders,
        "get_order_detail": _get_order_detail,
        "generate_order_card": _generate_order_card,
        "get_cart": _get_cart,
        "update_cart": _update_cart,
        "remove_cart": _remove_cart,
        "track_logistics": _track_logistics,
        "pay_order": _pay_order,
        "confirm_receive": _confirm_receive,
        "request_refund": _request_refund,
        "check_refund_eligibility": _check_refund_eligibility,
        "get_refund_info": _get_refund_info,
        "get_reviews": _get_reviews,
        "add_review": _add_review,
        "register_user": _register_user,
    }
    return handlers.get(tool_name)


# ══════════════════════════════════════════════════════════
#  用户身份
# ══════════════════════════════════════════════════════════

async def _lookup_user(args: dict) -> dict:
    """根据手机号查用户ID。"""
    phone = args.get("phone", "").strip()
    if not phone:
        return {"success": False, "error": "手机号不能为空"}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"http://localhost:8083/user/lookup", params={"phone": phone})
            data = resp.json()
            if data.get("success"):
                return {"success": True, "user_id": data["data"]["id"], "username": data["data"]["username"]}
            return {"success": False, "error": data.get("message", "未找到")}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  地址工具
# ══════════════════════════════════════════════════════════

async def _get_user_address(args: dict) -> dict:
    """调用 order-service 获取用户地址。"""
    user_id = args.get("user_id")
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{ORDER_SERVICE_URL}/api/address",
                params={"userId": user_id},
            )
            resp.raise_for_status()
            data = resp.json()
            addrs = data.get("data", []) if data.get("success") else []
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"get_user_address(user_id={user_id}) → {len(addrs)}条, {elapsed:.0f}ms")
            return {"addresses": addrs, "has_default": any(a.get("isDefault") for a in addrs)}
    except Exception as e:
        logger.warning(f"get_user_address 失败: {e}")
        return {"addresses": [], "has_default": False, "error": str(e)}


async def _save_user_address(args: dict) -> dict:
    """调用 order-service 保存地址。"""
    t0 = time.perf_counter()
    payload = {
        "userId": args.get("user_id"),
        "recipient": args.get("recipient"),
        "phone": args.get("phone"),
        "address": args.get("address"),
        "isDefault": args.get("is_default", True),
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{ORDER_SERVICE_URL}/api/address",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000
            if data.get("success"):
                logger.info(f"save_user_address → 成功, {elapsed:.0f}ms")
                return {"success": True, "address_id": data["data"]["id"]}
            return {"success": False, "error": data.get("message", "保存失败")}
    except Exception as e:
        logger.warning(f"save_user_address 失败: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  下单工具
# ══════════════════════════════════════════════════════════

async def _create_order(args: dict) -> dict:
    """
    将商品加入购物车。客户后续自己去购物车页面结算。
    参数全由 LLM 从上下文动态填入。
    """
    user_id = args.get("user_id")
    product_id = args.get("product_id")
    quantity = args.get("quantity", 1)
    category = args.get("category")

    missing = []
    if not user_id: missing.append("user_id")
    if not product_id: missing.append("product_id")
    if not category: missing.append("category")
    if missing:
        return {"success": False, "error": f"缺少必要参数: {', '.join(missing)}"}

    t0 = time.perf_counter()
    payload = {
        "userId": user_id,
        "productId": product_id,
        "quantity": quantity,
        "category": category,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{ORDER_SERVICE_URL}/api/cart/add",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"Agent加购 → user_id={user_id}, product_id={product_id}×{quantity}, {elapsed:.0f}ms")
            return {"success": True, "message": "已加入购物车，请客户前往商城购物车页面结算"}
    except Exception as e:
        logger.warning(f"Agent加购失败: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  图片识别
# ══════════════════════════════════════════════════════════

async def _analyze_image(args: dict) -> dict:
    """调用 Qwen-VL 分析图片"""
    from execution.vision_service import analyze_image
    image_url = args.get("image_url", "")
    question = args.get("question", "")
    if not image_url:
        return {"success": False, "error": "请提供图片URL"}
    try:
        result = await analyze_image(image_url, question)
        return {"success": True, "analysis": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  订单查询
# ══════════════════════════════════════════════════════════

async def _get_orders(args: dict) -> dict:
    """调用 order-service 查询用户订单列表"""
    user_id = args.get("user_id")
    if not user_id:
        return {"success": False, "error": "缺少 user_id"}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            url = f"{ORDER_SERVICE_URL}/api/order"
            logger.info(f"get_orders → GET {url}?userId={user_id}")
            resp = await client.get(url, params={"userId": user_id})
            elapsed = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"get_orders(user_id={user_id}) → 响应: {str(data)[:300]}, {elapsed:.0f}ms")
                # 宽松解析
                orders = data.get("data") or data.get("orders") or data.get("list") or []
                if isinstance(orders, list):
                    return {"success": True, "orders": orders, "total": len(orders)}
                return {"success": True, "orders": [], "total": 0}
            logger.warning(f"get_orders(user_id={user_id}) → HTTP {resp.status_code}, {elapsed:.0f}ms")
            return {"success": False, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        logger.error(f"get_orders(user_id={user_id}) 异常: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  订单详情
# ══════════════════════════════════════════════════════════

async def _get_order_detail(args: dict) -> dict:
    """调用 order-service 查询单个订单详情"""
    order_id = args.get("order_id")
    if not order_id:
        return {"success": False, "error": "缺少 order_id"}
    # 确保 order_id 是整数
    try:
        order_id = int(order_id)
    except (ValueError, TypeError):
        return {"success": False, "error": f"无效的订单ID: {order_id}"}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            url = f"{ORDER_SERVICE_URL}/api/order/{order_id}"
            logger.info(f"get_order_detail → GET {url}")
            resp = await client.get(url)
            elapsed = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"get_order_detail(order_id={order_id}) → 响应: {str(data)[:300]}, {elapsed:.0f}ms")
                # 宽松解析：有 data 字段就返回
                order_data = data.get("data") or data.get("order") or data
                if order_data and isinstance(order_data, dict):
                    return {"success": True, "order": order_data}
                # 如果没有嵌套，但响应本身包含订单字段
                if data.get("id") or data.get("total_amount") or data.get("status") is not None:
                    return {"success": True, "order": data}
                return {"success": False, "error": f"订单 {order_id} 未找到或响应异常: {str(data)[:200]}"}
            logger.warning(f"get_order_detail(order_id={order_id}) → HTTP {resp.status_code}, {elapsed:.0f}ms")
            return {"success": False, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        logger.error(f"get_order_detail(order_id={order_id}) 异常: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  订单卡片
# ══════════════════════════════════════════════════════════

async def _generate_order_card(args: dict) -> dict:
    """调用 order-service 生成订单卡片图片"""
    order_id = args.get("order_id")
    if not order_id:
        return {"success": False, "error": "请提供订单ID"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{ORDER_SERVICE_URL}/api/order/{order_id}/card"
            )
            if resp.status_code == 200:
                return {
                    "success": True,
                    "card_url": f"{ORDER_SERVICE_URL}/api/order/{order_id}/card",
                    "message": "订单卡片已生成"
                }
            return {"success": False, "error": f"订单服务返回 {resp.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  购物车查看
# ══════════════════════════════════════════════════════════

async def _get_cart(args: dict) -> dict:
    """调用 order-service 获取用户购物车内容"""
    user_id = args.get("user_id")
    if not user_id:
        return {"success": False, "error": "缺少 user_id"}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{ORDER_SERVICE_URL}/api/cart",
                params={"userId": user_id},
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000
            items = data.get("data", []) if data.get("success") else []
            logger.info(f"get_cart(user_id={user_id}) → {len(items)}条, {elapsed:.0f}ms")
            return {"success": True, "items": items, "total_items": len(items)}
    except Exception as e:
        logger.warning(f"get_cart 失败: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  购物车修改数量
# ══════════════════════════════════════════════════════════

async def _update_cart(args: dict) -> dict:
    """调用 order-service 修改购物车商品数量"""
    cart_id = args.get("cart_id")
    quantity = args.get("quantity")
    if not cart_id:
        return {"success": False, "error": "缺少 cart_id"}
    if not quantity or quantity < 1:
        return {"success": False, "error": "数量必须大于0"}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.put(
                f"{ORDER_SERVICE_URL}/api/cart/{cart_id}/quantity",
                json={"quantity": quantity},
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000
            if data.get("success"):
                logger.info(f"update_cart(cart_id={cart_id}, qty={quantity}) → 成功, {elapsed:.0f}ms")
                return {"success": True, "message": f"数量已更新为 {quantity}"}
            return {"success": False, "error": data.get("message", "更新失败")}
    except Exception as e:
        logger.warning(f"update_cart 失败: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  购物车删除商品
# ══════════════════════════════════════════════════════════

async def _remove_cart(args: dict) -> dict:
    """调用 order-service 删除购物车商品"""
    cart_id = args.get("cart_id")
    if not cart_id:
        return {"success": False, "error": "缺少 cart_id"}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.delete(
                f"{ORDER_SERVICE_URL}/api/cart/{cart_id}"
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000
            if data.get("success"):
                logger.info(f"remove_cart(cart_id={cart_id}) → 成功, {elapsed:.0f}ms")
                return {"success": True, "message": "已从购物车移除"}
            return {"success": False, "error": data.get("message", "删除失败")}
    except Exception as e:
        logger.warning(f"remove_cart 失败: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  物流追踪
# ══════════════════════════════════════════════════════════

async def _track_logistics(args: dict) -> dict:
    """调用 order-service 查询订单物流时间线"""
    order_id = args.get("order_id")
    if not order_id:
        return {"success": False, "error": "请提供订单ID"}
    try:
        order_id = int(order_id)
    except (ValueError, TypeError):
        return {"success": False, "error": f"无效的订单ID: {order_id}"}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{ORDER_SERVICE_URL}/api/order/logistics/{order_id}"
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"track_logistics(order_id={order_id}) → {elapsed:.0f}ms")
            if data.get("success"):
                logistics = data.get("data", {})
                return {"success": True, "logistics": logistics}
            return {"success": False, "error": data.get("message", "物流信息获取失败")}
    except Exception as e:
        logger.warning(f"track_logistics 失败: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  支付结算
# ══════════════════════════════════════════════════════════

async def _pay_order(args: dict) -> dict:
    """调用 pay-service 模拟支付。"""
    order_id = args.get("order_id")
    amount = args.get("amount")
    user_id = args.get("user_id")
    if not order_id:
        return {"success": False, "error": "缺少 order_id"}
    if not amount:
        return {"success": False, "error": "缺少支付金额 amount"}
    t0 = time.perf_counter()
    payload = {"orderId": order_id, "amount": amount}
    if user_id:
        payload["userId"] = user_id
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{PAY_SERVICE_URL}/api/pay/checkout",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"pay_order(order_id={order_id}, amount={amount}) → {elapsed:.0f}ms")
            if data.get("success"):
                return {"success": True, "message": f"订单 {order_id} 支付成功，金额 ¥{amount}，已进入发货流程"}
            return {"success": False, "error": data.get("message", "支付失败")}
    except Exception as e:
        logger.warning(f"pay_order 失败: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  确认收货
# ══════════════════════════════════════════════════════════

async def _confirm_receive(args: dict) -> dict:
    """调用 order-service 确认收货。"""
    order_id = args.get("order_id")
    if not order_id:
        return {"success": False, "error": "缺少 order_id"}
    try:
        order_id = int(order_id)
    except (ValueError, TypeError):
        return {"success": False, "error": f"无效的订单ID: {order_id}"}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.put(
                f"{ORDER_SERVICE_URL}/api/order/{order_id}/receive"
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"confirm_receive(order_id={order_id}) → {elapsed:.0f}ms")
            if data.get("success"):
                return {"success": True, "message": f"订单 {order_id} 已确认签收"}
            return {"success": False, "error": data.get("message", "确认收货失败")}
    except Exception as e:
        logger.warning(f"confirm_receive 失败: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  退款/售后
# ══════════════════════════════════════════════════════════

async def _request_refund(args: dict) -> dict:
    """调用 order-service 一键退货退款（申请+审核+退款自动完成）。"""
    order_id = args.get("order_id")
    reason = args.get("reason", "")
    if not order_id:
        return {"success": False, "error": "缺少 order_id"}
    if not reason:
        return {"success": False, "error": "请说明退货原因"}
    try:
        order_id = int(order_id)
    except (ValueError, TypeError):
        return {"success": False, "error": f"无效的订单ID: {order_id}"}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{ORDER_SERVICE_URL}/api/order/{order_id}/return/oneclick",
                json={"userId": args.get("user_id", 0), "reason": reason},
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"request_refund(order_id={order_id}, reason={reason}) → {elapsed:.0f}ms")
            if data.get("success"):
                result = data.get("data", {})
                order_sn = result.get("orderSn", "")
                refund_amount = result.get("refundAmount", "")
                msg = f"退货退款已完成！"
                if order_sn:
                    msg += f" 订单号：{order_sn}"
                if refund_amount:
                    msg += f"，退款金额：¥{refund_amount}"
                msg += f"。退款将在1-3个工作日到账。"
                return {"success": True, "message": msg, "orderSn": order_sn, "refundAmount": refund_amount}
            return {"success": False, "error": data.get("message", "退货退款失败")}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            data = e.response.json()
            return {"success": False, "error": data.get("message", "该订单不支持退货")}
        logger.warning(f"request_refund HTTP错误: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.warning(f"request_refund 失败: {e}")
        return {"success": False, "error": str(e)}


async def _check_refund_eligibility(args: dict) -> dict:
    """检查订单是否符合退货条件。"""
    order_id = args.get("order_id")
    if not order_id:
        return {"success": False, "error": "缺少 order_id"}
    try:
        order_id = int(order_id)
    except (ValueError, TypeError):
        return {"success": False, "error": f"无效的订单ID: {order_id}"}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{ORDER_SERVICE_URL}/api/order/{order_id}/return/check",
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"check_refund_eligibility(order_id={order_id}) → {elapsed:.0f}ms")
            if data.get("success"):
                result = data.get("data", {})
                eligible = result.get("eligible", False)
                if eligible:
                    return {
                        "success": True,
                        "eligible": True,
                        "orderId": result.get("orderId"),
                        "orderSn": result.get("orderSn", ""),
                        "orderStatus": result.get("orderStatus", ""),
                        "totalAmount": result.get("totalAmount", ""),
                        "message": f"订单符合退货条件。订单号：{result.get('orderSn', '')}，金额：¥{result.get('totalAmount', '')}",
                    }
                else:
                    return {
                        "success": True,
                        "eligible": False,
                        "message": result.get("reason", "不符合退货条件"),
                    }
            return {"success": False, "error": data.get("message", "查询失败")}
    except Exception as e:
        logger.warning(f"check_refund_eligibility 失败: {e}")
        return {"success": False, "error": str(e)}


async def _get_refund_info(args: dict) -> dict:
    """查询订单的退货进度和退款状态。"""
    order_id = args.get("order_id")
    if not order_id:
        return {"success": False, "error": "缺少 order_id"}
    try:
        order_id = int(order_id)
    except (ValueError, TypeError):
        return {"success": False, "error": f"无效的订单ID: {order_id}"}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{ORDER_SERVICE_URL}/api/order/{order_id}/return/info",
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"get_refund_info(order_id={order_id}) → {elapsed:.0f}ms")
            if data.get("success"):
                result = data.get("data", {})
                refund_status_text = result.get("refundStatusText", "无退货")
                order_sn = result.get("orderSn", "")
                total_amount = result.get("totalAmount", "")
                reason = result.get("reason", "")
                msg = f"订单 {order_sn} 退货状态：{refund_status_text}"
                if total_amount:
                    msg += f"，订单金额：¥{total_amount}"
                if reason:
                    msg += f"，退货原因：{reason}"
                return {
                    "success": True,
                    "message": msg,
                    "orderSn": order_sn,
                    "refundStatus": result.get("refundStatus"),
                    "refundStatusText": refund_status_text,
                    "totalAmount": total_amount,
                    "reason": reason,
                }
            return {"success": False, "error": data.get("message", "查询失败")}
    except Exception as e:
        logger.warning(f"get_refund_info 失败: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  商品评价
# ══════════════════════════════════════════════════════════

async def _get_reviews(args: dict) -> dict:
    """调用 product-service 查询商品评价。"""
    product_id = args.get("product_id")
    if not product_id:
        return {"success": False, "error": "缺少 product_id"}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{PRODUCT_API_URL.rsplit('/api/', 1)[0]}/api/review/product/{product_id}"
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000
            reviews = data.get("data", []) if data.get("success") else []
            logger.info(f"get_reviews(product_id={product_id}) → {len(reviews)}条, {elapsed:.0f}ms")
            if reviews:
                avg_rating = sum(r.get("rating", 0) for r in reviews) / len(reviews)
                return {
                    "success": True,
                    "reviews": reviews,
                    "total": len(reviews),
                    "avg_rating": round(avg_rating, 1),
                }
            return {"success": True, "reviews": [], "total": 0, "message": "该商品暂无评价"}
    except Exception as e:
        logger.warning(f"get_reviews 失败: {e}")
        return {"success": False, "error": str(e)}


async def _add_review(args: dict) -> dict:
    """调用 product-service 提交商品评价。"""
    product_id = args.get("product_id")
    user_id = args.get("user_id")
    rating = args.get("rating")
    comment = args.get("comment", "")
    missing = []
    if not product_id: missing.append("product_id")
    if not user_id: missing.append("user_id")
    if not rating: missing.append("rating")
    if not comment: missing.append("comment")
    if missing:
        return {"success": False, "error": f"缺少必要参数: {', '.join(missing)}"}
    if rating < 1 or rating > 5:
        return {"success": False, "error": "评分必须在1-5之间"}
    t0 = time.perf_counter()
    payload = {
        "productId": product_id,
        "userId": user_id,
        "rating": rating,
        "comment": comment,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{PRODUCT_API_URL.rsplit('/api/', 1)[0]}/api/review",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"add_review(product_id={product_id}, rating={rating}) → {elapsed:.0f}ms")
            if data.get("success"):
                return {"success": True, "message": f"评价已提交！感谢您的 {rating} 星评价 "}
            return {"success": False, "error": data.get("message", "评价提交失败")}
    except Exception as e:
        logger.warning(f"add_review 失败: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  用户注册
# ══════════════════════════════════════════════════════════

async def _register_user(args: dict) -> dict:
    """调用 user-service 注册新用户。"""
    username = args.get("username", "").strip()
    phone = args.get("phone", "").strip()
    password = args.get("password", "").strip()
    missing = []
    if not username: missing.append("用户名")
    if not phone: missing.append("手机号")
    if not password: missing.append("密码")
    if missing:
        return {"success": False, "error": f"缺少必要信息: {', '.join(missing)}"}
    t0 = time.perf_counter()
    payload = {
        "username": username,
        "phone": phone,
        "password": password,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{USER_SERVICE_URL}/api/user/register",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"register_user(username={username}, phone={phone}) → {elapsed:.0f}ms")
            if data.get("success"):
                user_data = data.get("data", {})
                user_id = user_data.get("id", "未知")
                points = user_data.get("points", 100)
                return {
                    "success": True,
                    "message": f"注册成功！欢迎加入 PetHome 🎉 您的用户ID: {user_id}，已获得 {points} 积分",
                    "user_id": user_id,
                    "points": points,
                }
            return {"success": False, "error": data.get("message", "注册失败")}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 409:
            return {"success": False, "error": "该手机号已注册，请直接登录"}
        logger.warning(f"register_user HTTP错误: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.warning(f"register_user 失败: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  降级策略（工具失败后的兜底方案）
# ══════════════════════════════════════════════════════════

async def _degrade_search_product(args: dict, original_error: str) -> dict:
    """
    商品搜索降级：product-service 不可用时，从知识库搜索相关商品信息。
    """
    keyword = args.get("keyword", "")
    logger.info(f"[降级] search_product → 知识库检索: '{keyword}'")
    results = search_knowledge(keyword, top_k=5, use_hybrid=False, use_rewrite=False)
    if results:
        return {
            "success": True,
            "source": "knowledge_base_fallback",
            "message": f"商品服务暂不可用，从知识库为您找到以下信息：",
            "results": results,
            "warning": "当前结果来自知识库，非实时商品数据",
        }
    return {
        "success": False,
        "error": "商品服务暂不可用，知识库中也未找到相关信息，请稍后再试",
        "original_error": original_error,
    }


async def _degrade_get_orders(args: dict, original_error: str) -> dict:
    """订单查询降级：返回友好提示。"""
    return {
        "success": False,
        "error": "订单服务暂时不可用，请稍后再试或联系客服人工查询",
        "suggestion": "您可以直接前往商城的'我的订单'页面查看",
        "original_error": original_error,
    }


async def _degrade_get_order_detail(args: dict, original_error: str) -> dict:
    """订单详情降级。"""
    return {
        "success": False,
        "error": "订单详情服务暂时不可用",
        "suggestion": "请稍后再试，或前往商城'我的订单'页面查看",
        "original_error": original_error,
    }


async def _degrade_get_cart(args: dict, original_error: str) -> dict:
    """购物车查询降级。"""
    return {
        "success": False,
        "error": "购物车服务暂时不可用",
        "suggestion": "请直接前往商城购物车页面查看",
        "original_error": original_error,
    }


async def _degrade_track_logistics(args: dict, original_error: str) -> dict:
    """物流查询降级。"""
    return {
        "success": False,
        "error": "物流查询服务暂时不可用",
        "suggestion": "请稍后再试，或前往商城'我的订单'页面查看物流信息",
        "original_error": original_error,
    }
