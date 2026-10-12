"""
商城 API 服务 — 对标 Spring Boot @Service
包含登录认证 + 商品搜索（重试 + 字段过滤）。
"""

import asyncio
import json
import time

import httpx
from loguru import logger

from config import (
    PRODUCT_API_URL,
    PRODUCT_LOGIN_URL,
    PRODUCT_API_USERNAME,
    PRODUCT_API_PASSWORD,
    PRODUCT_API_TOKEN,
)

# ── 全局 token 缓存 ────────────────────────────────
_product_api_token: str | None = PRODUCT_API_TOKEN or None

# ── 搜索结果缓存（5分钟TTL）─────────────────────────
_search_cache: dict[str, tuple[dict, float]] = {}
CACHE_TTL = 300  # 秒


async def _login_product_api() -> str | None:
    """用配置的账号密码登录商城，返回 token；失败返回 None。"""
    if not PRODUCT_API_USERNAME or not PRODUCT_API_PASSWORD:
        logger.warning("未配置 PRODUCT_API_USERNAME / PRODUCT_API_PASSWORD，跳过登录")
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                PRODUCT_LOGIN_URL,
                json={"username": PRODUCT_API_USERNAME, "password": PRODUCT_API_PASSWORD, "type": 1},
            )
            resp.raise_for_status()
            data = resp.json()
            token = (data.get("data") or {}).get("token", "")
            if token:
                logger.info(f"商城登录成功，token 前8位: {token[:8]}...")
                return token
            logger.warning(f"商城登录失败，响应: {json.dumps(data, ensure_ascii=False)[:300]}")
    except Exception as e:
        logger.warning(f"商城登录失败: {e}")
    return None


async def get_product_token() -> str | None:
    """获取产品 API token（缓存复用，失效时自动刷新）。"""
    global _product_api_token
    if _product_api_token:
        return _product_api_token
    _product_api_token = await _login_product_api()
    return _product_api_token


def _extract_products(data) -> list[dict]:
    """递归/多路径提取商品列表，适配各种 API 返回结构。"""
    if isinstance(data, list):
        return data if all(isinstance(x, dict) for x in data) else []
    if not isinstance(data, dict):
        return []
    for key in ("data", "records", "rows", "products", "list", "result"):
        if key in data:
            val = data[key]
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                inner = _extract_products(val)
                if inner:
                    return inner
    return []


async def search_product_api(args: dict) -> dict:
    """
    异步调用商城搜索接口，超时 5s，最多重试 3 次（指数退避）。
    自动携带 token，noLogin 时自动重新登录，失败返回空列表。
    内置 5 分钟内存缓存 + 品种→体型智能映射。

    Args:
        args: {"keyword": str, "page": int, "size": int}
    """
    keyword = args.get("keyword", "")
    page = args.get("page", 1)
    size = args.get("size", 10)
    # ── 品种→体型映射（搜不到时自动降级）──────
    result = await _search_product_raw(keyword, page, size)
    if result["total"] > 0:
        return result

    # 搜不到 → 品种映射为体型重试
    mapped = _map_breed_to_size(keyword)
    if mapped and mapped != keyword:
        logger.info(f"[智能映射] '{keyword}' → '{mapped}' 重试搜索")
        result = await _search_product_raw(mapped, page, size)
        if result["total"] > 0:
            return result

    # 还是搜不到 → 去掉品种名，用通用词兜底
    fallback = _fallback_keyword(keyword)
    if fallback and fallback not in (keyword, mapped):
        logger.info(f"[兜底搜索] '{keyword}' → '{fallback}'")
        result = await _search_product_raw(fallback, page, size)

    return result


# ── 品种→体型映射表 ─────────────────────────────
BREED_TO_SIZE = {
    # 大型犬
    "哈士奇": "大型犬", "金毛": "大型犬", "拉布拉多": "大型犬",
    "阿拉斯加": "大型犬", "德牧": "大型犬", "萨摩耶": "大型犬",
    "边牧": "大型犬", "边境牧羊犬": "大型犬", "罗威纳": "大型犬",
    "大白熊": "大型犬", "圣伯纳": "大型犬", "秋田": "大型犬",
    "柴犬": "中型犬", "柯基": "中型犬", "法斗": "小型犬",
    # 小型犬
    "泰迪": "小型犬", "比熊": "小型犬", "博美": "小型犬",
    "吉娃娃": "小型犬", "约克夏": "小型犬", "雪纳瑞": "小型犬",
    "贵宾": "小型犬", "马尔济斯": "小型犬", "巴哥": "小型犬",
    "蝴蝶犬": "小型犬", "西施": "小型犬", "腊肠": "小型犬",
    # 猫品种 → 猫粮
    "布偶": "猫粮", "英短": "猫粮", "美短": "猫粮",
    "暹罗": "猫粮", "波斯": "猫粮", "无毛": "猫粮",
    "缅因": "猫粮", "橘猫": "猫粮", "狸花": "猫粮",
    "蓝猫": "猫粮", "金渐层": "猫粮", "银渐层": "猫粮",
    "加菲": "猫粮", "异国短毛猫": "猫粮", "折耳": "猫粮",
    "苏格兰折耳": "猫粮", "德文": "猫粮", "阿比西尼亚": "猫粮",
    "孟加拉": "猫粮", "挪威森林猫": "猫粮", "西伯利亚": "猫粮",
    "金吉拉": "猫粮", "曼基康": "猫粮", "东短": "猫粮",
    "东方短毛猫": "猫粮", "三花": "猫粮", "奶牛猫": "猫粮",
    "玳瑁": "猫粮", "乳白": "猫粮", "中华田园猫": "猫粮",
    # 猫咪特殊需求
    "幼猫": "猫粮 幼猫", "小猫": "猫粮 幼猫",
    "老猫": "猫粮 老年猫", "老年猫": "猫粮 老年猫",
    # 狗狗特殊需求
    "幼犬": "狗粮 幼犬", "小狗": "狗粮 幼犬",
    "老狗": "狗粮 老年犬", "老年犬": "狗粮 老年犬",
}


def _map_breed_to_size(keyword: str) -> str | None:
    """如果关键词含品种名，替换为对应体型/分类。"""
    for breed, size in BREED_TO_SIZE.items():
        if breed in keyword:
            # 替换品种名为体型，保留其他词
            return keyword.replace(breed, size).strip()
    return None


def _fallback_keyword(keyword: str) -> str | None:
    """搜不到时降级为通用关键词。"""
    if any(w in keyword for w in ("犬", "狗", "柯基", "金毛", "柴犬", "泰迪", "拉布")):
        return "狗粮"
    if any(w in keyword for w in ("猫", "布偶", "英短", "美短", "暹罗", "缅因")):
        return "猫粮"
    if "零食" in keyword or "冻干" in keyword or "罐头" in keyword:
        return "零食"
    return None


async def _search_product_raw(keyword: str, page: int = 1, size: int = 10) -> dict:
    """
    实际 HTTP 调用（不带映射逻辑，被 search_product_api 包装）。
    """
    # ── 缓存检查 ──
    cache_key = f"{keyword}:{page}:{size}"
    if cache_key in _search_cache:
        cached_result, cached_ts = _search_cache[cache_key]
        if time.time() - cached_ts < CACHE_TTL:
            logger.info(f"[缓存命中] search_product(keyword={keyword}) → {cached_result['total']}条")
            return cached_result
        del _search_cache[cache_key]

    t0 = time.perf_counter()
    last_error = ""
    keep_fields = {"id", "name", "price", "brand", "weight", "description"}

    token = await get_product_token()

    async with httpx.AsyncClient(timeout=5.0) as client:
        for attempt in range(3):
            try:
                headers = {"token": token} if token else {}
                resp = await client.get(
                    PRODUCT_API_URL,
                    params={"keyword": keyword, "page": page, "size": size},
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

                # noLogin → 清缓存重登录
                if isinstance(data, dict) and data.get("message") == "noLogin":
                    logger.warning("token 失效，尝试重新登录...")
                    global _product_api_token
                    _product_api_token = None
                    token = await get_product_token()
                    if token:
                        logger.info("重登录成功，用新 token 重试")
                        continue
                    last_error = "noLogin（重登录失败）"
                    break

                raw_products = _extract_products(data)
                products = [
                    {k: v for k, v in p.items() if k in keep_fields}
                    for p in raw_products
                ]

                elapsed_ms = (time.perf_counter() - t0) * 1000
                if products:
                    logger.info(
                        f"search_product(keyword={keyword}) → "
                        f"返回{len(products)}条, 耗时{elapsed_ms:.0f}ms"
                    )
                else:
                    raw_snippet = json.dumps(data, ensure_ascii=False)[:500]
                    logger.warning(
                        f"search_product(keyword={keyword}) → 解析为0条!\n"
                        f"  原始响应: {raw_snippet}"
                    )
                result = {"keyword": keyword, "total": len(products), "products": products}
                _search_cache[cache_key] = (result, time.time())
                return result

            except httpx.TimeoutException:
                last_error = "timeout"
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}"
            except Exception as e:
                last_error = str(e)

            if attempt < 2:
                await asyncio.sleep(2 ** attempt)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.warning(
        f"search_product(keyword={keyword}) → 失败({last_error}), "
        f"已重试3次, 耗时{elapsed_ms:.0f}ms, 返回空列表"
    )
    return {"keyword": keyword, "total": 0, "products": []}
