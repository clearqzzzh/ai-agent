"""
工具定义 — Function Calling 注册表
===================================
所有业务规则已迁移到 config/system_prompt.py，本文件只保留纯技术定义：
  - 工具名、功能一句话描述
  - 入参 schema（名称、类型、是否必填）
  - 每个参数的简短说明
"""

TOOLS = [
    # ══════════════════════════════════════════════════════
    #  商品搜索
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "search_product",
            "description": "在宠物商城搜索商品。传入搜索关键词，返回匹配的商品列表（id、名称、价格、品牌、规格、描述）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词",
                    }
                },
                "required": ["keyword"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  喂养知识库
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索宠物喂养专业知识库（RAG），涵盖猫狗的饮食禁忌、营养搭配、日常护理等内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户的问题或搜索关键词",
                    }
                },
                "required": ["query"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  医疗知识库
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "search_medical_knowledge",
            "description": "搜索宠物医疗健康知识库（RAG），涵盖常见病症识别、家庭护理方法、就医指引、情绪安抚等内容。当用户描述宠物症状、健康问题或表达担忧时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户的症状描述或健康问题",
                    }
                },
                "required": ["query"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  用户身份
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "lookup_user",
            "description": "根据手机号查询商城用户信息，返回 user_id 和用户名。",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {
                        "type": "string",
                        "description": "用户手机号",
                    }
                },
                "required": ["phone"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  收货地址
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "get_user_address",
            "description": "获取用户的收货地址列表，返回地址数组（含收件人、电话、详细地址、是否默认）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "用户ID，从 system prompt 末尾的 [当前登录用户ID: N] 中获取",
                    }
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_user_address",
            "description": "保存新的收货地址到用户地址簿。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "用户ID",
                    },
                    "recipient": {
                        "type": "string",
                        "description": "收件人姓名",
                    },
                    "phone": {
                        "type": "string",
                        "description": "联系电话",
                    },
                    "address": {
                        "type": "string",
                        "description": "详细收货地址",
                    },
                    "is_default": {
                        "type": "boolean",
                        "description": "是否设为默认地址，通常为 true",
                    },
                },
                "required": ["user_id", "recipient", "phone", "address"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  加购
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": "将商品加入购物车。返回后端计算的 total_amount，直接展示，不要心算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "用户ID",
                    },
                    "product_id": {
                        "type": "integer",
                        "description": "商品ID，从 search_product 结果中提取",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "购买数量，用户没说则默认 1",
                    },
                    "category": {
                        "type": "string",
                        "description": "商品分类（猫粮/狗粮/零食/用品），从 search_product 结果中提取",
                    },
                    "recipient": {
                        "type": "string",
                        "description": "收货人姓名（可选，从地址信息中获取）",
                    },
                    "phone": {
                        "type": "string",
                        "description": "联系电话（可选，从地址信息中获取）",
                    },
                    "address": {
                        "type": "string",
                        "description": "收货地址（可选，从地址信息中获取）",
                    },
                },
                "required": ["user_id", "product_id", "category"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  图片识别（Qwen-VL）
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": (
                "分析用户上传的图片内容。用于识别宠物品种、猫粮品牌和配料、宠物症状等。"
                "当用户发送图片或提到拍照/上传时调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {
                        "type": "string",
                        "description": "用户上传的图片URL或base64 data URI",
                    },
                    "question": {
                        "type": "string",
                        "description": "用户对图片的具体问题，如'这是什么品种'、'这个猫粮成分怎么样'",
                    },
                },
                "required": ["image_url"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  订单查询
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "get_orders",
            "description": "查询用户的订单列表。用户说'我的订单''查订单'时调用。user_id 从 system prompt 末尾获取，不需要用户提供。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "用户ID，从system prompt末尾的[当前登录用户ID: N]获取",
                    },
                },
                "required": ["user_id"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  订单详情
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "get_order_detail",
            "description": "根据订单ID查询单个订单的详细信息（商品、金额、状态、物流等）。订单ID是订单表的主键id，用户可能称之为订单号。当用户发送了一个数字ID或提到订单号时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "订单ID，从用户消息中提取",
                    },
                },
                "required": ["order_id"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  订单卡片
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "generate_order_card",
            "description": (
                "生成订单卡片图片。当用户说'我的订单'、'订单到哪了'、'帮我查一下订单'时调用。"
                "需要从对话上下文中提取 order_id。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "订单ID，从对话历史或用户消息中提取",
                    },
                },
                "required": ["order_id"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  购物车查看
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "get_cart",
            "description": "查看用户的购物车内容。返回购物车中所有商品的名称、单价、数量、小计金额。用户说'看看购物车''购物车里有啥'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "用户ID，从system prompt末尾的[当前登录用户ID: N]获取",
                    },
                },
                "required": ["user_id"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  购物车修改数量
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "update_cart",
            "description": "修改购物车中某个商品的数量。用户说'把XX改成2件''数量改成3'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "cart_id": {
                        "type": "integer",
                        "description": "购物车条目ID，从 get_cart 返回结果中的 id 字段获取",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "新的数量，必须大于0",
                    },
                },
                "required": ["cart_id", "quantity"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  购物车删除商品
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "remove_cart",
            "description": "从购物车中删除某个商品。用户说'把XX删了''不要这个了'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "cart_id": {
                        "type": "integer",
                        "description": "购物车条目ID，从 get_cart 返回结果中的 id 字段获取",
                    },
                },
                "required": ["cart_id"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  物流追踪
    # ═════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "track_logistics",
            "description": "查询订单的物流信息和配送时间线。用户说'快递到哪了''物流怎么样''我的包裹呢''发货了吗'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "订单ID，从对话历史或用户消息中提取",
                    },
                },
                "required": ["order_id"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  支付结算
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "pay_order",
            "description": "对订单进行模拟支付。用户说'付款''结账''支付订单'时调用。支付成功后订单会自动进入发货流程。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "订单ID",
                    },
                    "amount": {
                        "type": "number",
                        "description": "支付金额，从订单详情中获取 total_amount",
                    },
                    "user_id": {
                        "type": "integer",
                        "description": "用户ID，从system prompt末尾获取",
                    },
                },
                "required": ["order_id", "amount"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  确认收货
    # ═════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "confirm_receive",
            "description": "确认订单已到货。用户说'确认收货''已收到货'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "订单ID",
                    },
                },
                "required": ["order_id"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  退款/售后
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "check_refund_eligibility",
            "description": "检查订单是否符合退货条件。用户说'我要退货''能不能退'时先调用此工具确认资格。返回是否可退、订单状态、金额等信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "订单ID，从对话历史或用户消息中提取",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_refund",
            "description": "一键退货退款（申请+审核+退款自动完成）。确认退货资格后，用户同意退货时调用。待发货/运输中/已到货的订单均可退。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "订单ID",
                    },
                    "reason": {
                        "type": "string",
                        "description": "退货原因，如'商品质量问题''不想要了''发错货''宠物不喜欢'",
                    },
                },
                "required": ["order_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_refund_info",
            "description": "查询订单的退货进度和退款状态。用户问'退货进度怎样了''退款到了吗''退货审核了吗'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "订单ID",
                    },
                },
                "required": ["order_id"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  商品评价
    # ══════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "get_reviews",
            "description": "查询某商品的用户评价列表。用户问'这个商品评价怎么样''有人买过吗''好评多吗'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "商品ID，从 search_product 结果或对话上下文中获取",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_review",
            "description": "为已到货的订单商品提交评价。用户说'我要评价''给个好评''写评论'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "商品ID",
                    },
                    "user_id": {
                        "type": "integer",
                        "description": "用户ID，从system prompt末尾获取",
                    },
                    "rating": {
                        "type": "integer",
                        "description": "评分，1-5星",
                    },
                    "comment": {
                        "type": "string",
                        "description": "评价内容",
                    },
                },
                "required": ["product_id", "user_id", "rating", "comment"],
            },
        },
    },

    # ══════════════════════════════════════════════════════
    #  用户注册
    # ═════════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "register_user",
            "description": "注册新用户账号。用户说'我要注册''创建一个账号''新用户'时调用。注册成功后自动获得100积分。",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "用户名/昵称",
                    },
                    "phone": {
                        "type": "string",
                        "description": "手机号，用于登录",
                    },
                    "password": {
                        "type": "string",
                        "description": "登录密码",
                    },
                },
                "required": ["username", "phone", "password"],
            },
        },
    },
]
