"""
AI Agent 端到端压测脚本 (Locust)

用法:
  1. pip install locust
  2. cd 到本文件所在目录
  3. 启动 Web 界面:  locust -f locustfile.py
  4. 浏览器打开 http://localhost:8089
  5. 填入并发用户数、每秒新增用户数、目标地址 http://localhost:8000
  6. 点击 Start

无界面模式:
  locust -f locustfile.py --headless -u 50 -r 5 --run-time 60s --host http://localhost:8000

参数说明:
  -u 50    总共 50 个并发用户
  -r 5     每秒启动 5 个用户
  --run-time 60s  压测持续 60 秒
"""

import json
import random
import string
import time
import uuid

from locust import HttpUser, task, between, events
from locust.runners import local


# ============================================================
# 模拟真实用户的对话语料库
# ============================================================

# 商品咨询类（会触发 search_product）
PRODUCT_QUERIES = [
    "推荐一款适合布偶猫的猫粮",
    "有没有 200 以内的狗粮",
    "我想买猫砂，有推荐吗",
    "皇家品牌的猫粮有哪些",
    "帮我看看宠物玩具",
    "有没有适合幼犬的零食",
    "推荐一款猫罐头",
    "我想买猫爬架，300以内",
    "有没有自动喂食器",
    "帮我找一下猫咪驱虫药",
]

# 售后咨询类（会触发 query_knowledge_base）
AFTERSALES_QUERIES = [
    "你们怎么退货啊",
    "保修期是多久",
    "物流怎么查",
    "发票什么时候开",
    "收到货坏了怎么办",
    "怎么申请退款",
    "运费谁承担",
    "换货流程是什么",
]

# 加购类（会触发 add_to_cart，需要 user_id）
ADD_TO_CART_QUERIES = [
    "帮我把第一个加入购物车",
    "我要买这个，加购",
    "把刚才推荐的加到购物车",
]

# 闲聊类（不触发工具调用）
CASUAL_QUERIES = [
    "你好",
    "在吗",
    "谢谢",
    "你们几点下班",
    "有人工客服吗",
]


class AiAgentUser(HttpUser):
    """
    模拟一个真实用户和 AI Agent 对话的行为。

    每个用户:
    - 有一个固定的 session_id（模拟多轮对话）
    - 随机 70% 概率带 user_id（模拟已登录）
    - 等待 2-8 秒后发下一条消息（模拟真人阅读和思考）
    """

    # 每次请求之间的等待时间（秒）
    # AI 接口响应慢（LLM 调用），所以等待时间长一些
    wait_time = between(2, 8)

    def on_start(self):
        """用户启动时初始化会话"""
        self.session_id = str(uuid.uuid4())
        self.user_id = random.randint(1, 100) if random.random() < 0.7 else None
        self.turn_count = 0
        self.max_turns = random.randint(3, 8)  # 每个用户最多聊 3-8 轮

    def _send_chat(self, message: str, tag: str = ""):
        """
        发送一条对话请求并记录自定义事件指标。

        tag: 用于在 Locust 统计中区分不同类型的请求
        """
        payload = {
            "messages": [{"role": "user", "content": message}],
            "session_id": self.session_id,
        }
        if self.user_id:
            payload["user_id"] = self.user_id

        start_time = time.perf_counter()
        try:
            with self.client.post(
                "/chat",
                json=payload,
                name=f"/chat [{tag}]",
                catch_response=True,
            ) as resp:
                elapsed = (time.perf_counter() - start_time) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    reply = data.get("reply", "")
                    tool_calls = data.get("tool_calls_made", [])

                    # 基本校验
                    if not reply:
                        resp.failure("回复为空")
                    elif len(reply) < 2:
                        resp.failure(f"回复过短: {reply!r}")
                    else:
                        # 记录额外指标
                        events.request.fire(
                            request_type="METRIC",
                            name=f"tool_calls [{tag}]",
                            response_time=elapsed,
                            response_length=len(tool_calls),
                            exception=None,
                            context={},
                        )
                        resp.success()
                else:
                    resp.failure(f"HTTP {resp.status_code}: {resp.text[:100]}")

        except Exception as e:
            events.request.fire(
                request_type="ERROR",
                name=f"exception [{tag}]",
                response_time=(time.perf_counter() - start_time) * 1000,
                response_length=0,
                exception=e,
                context={},
            )

    # ---- 按权重分配任务 ----

    @task(7)
    def ask_product(self):
        """商品咨询（权重 7，最高频）"""
        msg = random.choice(PRODUCT_QUERIES)
        self._send_chat(msg, tag="商品咨询")
        self.turn_count += 1

    @task(3)
    def ask_aftersales(self):
        """售后咨询（权重 3）"""
        msg = random.choice(AFTERSALES_QUERIES)
        self._send_chat(msg, tag="售后咨询")
        self.turn_count += 1

    @task(1)
    def ask_add_to_cart(self):
        """加购操作（权重 1，最低频）"""
        if not self.user_id:
            return  # 未登录用户跳过
        msg = random.choice(ADD_TO_CART_QUERIES)
        self._send_chat(msg, tag="加购")
        self.turn_count += 1

    @task(2)
    def casual_chat(self):
        """闲聊（权重 2）"""
        msg = random.choice(CASUAL_QUERIES)
        self._send_chat(msg, tag="闲聊")
        self.turn_count += 1

    @task(1)
    def multi_turn_conversation(self):
        """
        模拟多轮对话：先搜商品，再追问，再加购。
        一个 task 内连续发多条消息，模拟真实对话流。
        """
        if self.turn_count >= self.max_turns:
            return

        # 第一轮：搜商品
        self._send_chat("推荐一款猫粮", tag="多轮-搜索")
        self.turn_count += 1

        # 等一小下再发第二轮（模拟用户阅读结果）
        time.sleep(random.uniform(1, 3))

        # 第二轮：追问
        follow_ups = [
            "有没有便宜一点的",
            "第二个怎么样",
            "有试用装吗",
            "帮我看看评价",
        ]
        self._send_chat(random.choice(follow_ups), tag="多轮-追问")
        self.turn_count += 1

        # 第三轮：加购（如果已登录）
        if self.user_id and random.random() < 0.5:
            time.sleep(random.uniform(1, 2))
            self._send_chat("帮我加购第一个", tag="多轮-加购")
            self.turn_count += 1


# ============================================================
# 自定义统计报告（压测结束后打印）
# ============================================================

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """压测结束时打印关键指标"""
    stats = environment.stats
    total = stats.total

    print("\n" + "=" * 60)
    print("压测结果汇总")
    print("=" * 60)
    print(f"总请求数:    {total.num_requests}")
    print(f"失败请求数:  {total.num_failures}")
    print(f"失败率:      {total.fail_ratio * 100:.1f}%")
    print(f"平均响应:    {total.avg_response_time:.0f}ms")
    print(f"P50 响应:    {total.get_response_time_percentile(0.50):.0f}ms")
    print(f"P95 响应:    {total.get_response_time_percentile(0.95):.0f}ms")
    print(f"P99 响应:    {total.get_response_time_percentile(0.99):.0f}ms")
    print(f"RPS:         {total.current_rps:.1f}")
    print("=" * 60)

    # 按类型统计
    for entry in stats.entries.values():
        if entry.name.startswith("/chat"):
            print(f"\n[{entry.name}]")
            print(f"  请求数: {entry.num_requests}  失败率: {entry.fail_ratio * 100:.1f}%")
            print(f"  平均: {entry.avg_response_time:.0f}ms  P95: {entry.get_response_time_percentile(0.95):.0f}ms")
