"""各Agent专属工具集 — 从统一TOOLS中按业务功能拆分

7-Agent架构：
  1. 意图分析调度（无工具，纯路由）
  2. 商品咨询 ProductTools
  3. 下单操作 OrderTools
  4. 售后处理 AfterSaleTools
  5. 宠物健康 PetHealthTools
  6. 知识检索 RetrievalTools
  7. 未知意图兜底（无工具，纯兜底回复）
"""

from cognition.definitions import TOOLS

_tool_map = {t["function"]["name"]: t for t in TOOLS}

# ══════════════════════════════════════════════════════════════
#  商品咨询智能体 — ProductTools
#  商品搜索、图片识别、用户评价查看
# ══════════════════════════════════════════════════════════════
PRODUCT_CONSULT_TOOLS = [
    _tool_map["search_product"],
    _tool_map["analyze_image"],
    _tool_map["get_reviews"],
    _tool_map["add_review"],
]

# ══════════════════════════════════════════════════════════════
#  下单操作智能体 — OrderTools
#  地址管理、下单加购、购物车管理、订单查询、物流追踪、支付确认收货
# ══════════════════════════════════════════════════════════════
ORDER_OPS_TOOLS = [
    _tool_map["get_user_address"],
    _tool_map["save_user_address"],
    _tool_map["create_order"],
    _tool_map["lookup_user"],
    _tool_map["get_orders"],
    _tool_map["get_order_detail"],
    _tool_map["generate_order_card"],
    _tool_map["get_cart"],
    _tool_map["update_cart"],
    _tool_map["remove_cart"],
    _tool_map["track_logistics"],
    _tool_map["pay_order"],
    _tool_map["confirm_receive"],
    _tool_map["register_user"],
]

# ══════════════════════════════════════════════════════════════
#  售后处理智能体 — AfterSaleTools
#  退货资格检查、一键退货退款、退款进度查询
# ══════════════════════════════════════════════════════════════
AFTERSALE_TOOLS = [
    _tool_map["check_refund_eligibility"],
    _tool_map["request_refund"],
    _tool_map["get_refund_info"],
]

# ══════════════════════════════════════════════════════════════
#  宠物健康智能体 — PetHealthTools
#  医疗知识库检索、健康相关商品推荐
# ══════════════════════════════════════════════════════════════
PET_HEALTH_TOOLS = [
    _tool_map["search_medical_knowledge"],
    _tool_map["search_product"],
]

# ══════════════════════════════════════════════════════════════
#  知识检索智能体 — RetrievalTools
#  喂养知识库检索（通用知识问答）
# ══════════════════════════════════════════════════════════════
KNOWLEDGE_RETRIEVAL_TOOLS = [
    _tool_map["search_knowledge"],
]

# ══════════════════════════════════════════════════════════════
#  意图分析调度智能体 — 无工具（纯路由决策）
#  未知意图兜底智能体 — 无工具（纯兜底回复）
# ══════════════════════════════════════════════════════════════
INTENT_SCHEDULER_TOOLS: list = []
UNKNOWN_INTENT_TOOLS: list = []
