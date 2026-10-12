# Agent 性能优化记录

## 问题

首次上线时单次 `/chat` 请求耗时 **10-12 秒**，其中 LLM 调用占比约 80%。

## 根因

Supervisor 模式下每轮对话走了 **4 次 LLM 调用**：

```
用户消息
  → Supervisor(LLM #1, 1-2s) → 意图路由
  → Agent(LLM #2, 3-4s) → 决定调哪个工具 → 执行工具(0.5s)
  → Supervisor(LLM #3, 1-2s) → "还要继续吗？"
  → Agent(LLM #4, 3-4s) → 根据工具结果生成文本回复
  → Supervisor(LLM #5) → FINISH

总 LLM 耗时 ≈ 8-10s
```

每次 Agent 做完事后必须回 Supervisor 请示下一步，导致工具执行和文本回复之间多了一次往返。

## 优化方案

### V1 — Agent 内联生成回复（已实施）

Agent 执行完工具拿到结果后，**立即调第二次 LLM 生成文本回复**，不再回到 Supervisor 再绕一圈。

```
优化后（2次 LLM）：
  用户消息
    → Supervisor(LLM #1, 1-2s) → 意图路由
    → Agent:
        ├─ LLM #2 (3-4s) → 决定调工具
        ├─ 执行工具 (0.5s)
        └─ LLM #3 (3-4s) → 根据结果生成文本回复
    → Supervisor → FINISH

总 LLM 耗时 ≈ 7-9s（降 ~25%）
```

**改动的文件**：`orchestration/supervisor_graph.py` — `_agent_node()` 函数

### V2 — Supervisor 首轮规则匹配（待实施）

首轮对话不调 LLM，用关键词规则判断意图：

```python
def quick_intent(msg: str) -> str:
    if any(w in msg for w in ["推荐","有没有","狗粮","猫粮","零食","用品","吃什么","怎么喂"]):
        return "product_agent"
    if any(w in msg for w in ["下单","买","地址","购物车","结算"]):
        return "order_agent"
    return "FINISH"  # 闲聊直接回复
```

预期再省 1-2s。

### V3 — SSE 流式输出（待实施）

将 `/chat` 改为 Server-Sent Events，LLM 每生成一个 token 就推给前端。总耗时不变但**体感延迟趋近于 0**。

```python
# main.py
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        stream = _client.chat.completions.create(..., stream=True)
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'token': chunk.choices[0].delta.content})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## 效果对比

| 版本 | LLM 调用次数 | 预计延迟 | 状态 |
|------|:-----------:|---------|:----:|
| V0（最初） | 4 | 10-12s | 已废弃 |
| V1（内联回复） | 3 | 7-9s | ✅ 已实施 |
| V2（规则路由） | 2 | 5-7s | 待实施 |
| V3（SSE流式） | 2 | 首token < 2s | 待实施 |
| V4（摘要优化） | 3 | 8-10s | ✅ 已实施 |

---

## Bug 修复记录

### 🐛 会话存储丢失消息

**问题**：对话完成后 Redis 里消息数为 0，历史列表为空。

**根因**：`_trim_to_rounds()` off-by-one 错误。当轮数不超 MAX_ROUNDS 时，`cutoff` 初始值为 `len(messages)`，切片 `messages[len:]` 永远返回空数组。

**修复**：`memory/session_store.py` — 重写为从后往前数到第 `max_rounds+1` 个 user 消息时才截断，否则全保留。

### 🐛 历史会话列表缺失老会话

**问题**：从 JSON 文件迁移到 Redis 前的老会话在列表中不显示。

**根因**：`list_sessions` 只查 Redis ZSet 索引，老会话不在索引中。且索引回写逻辑会把删掉的会话又加回来。

**修复**：`list_sessions` 改为 ZSet + KEYS 双重扫描，不再自动回写索引。

### 🐛 侧边栏不显示

**问题**：Streamlit 侧边栏在宠物商城 iframe 中不可见。

**根因**：iframe 宽度 420px，Streamlit 自动折叠侧边栏（响应式断点）。

**修复**：历史会话切换从侧边栏移到主区域顶部 `st.expander` 折叠面板中。

### 🐛 Graph 中间消息丢失

**问题**：保存的会话只有最终文本回复，缺少工具调用和结果消息。

**根因**：`chat_service.py` 只追加 `final_reply` 到 history，graph 返回的 `result_messages` 被丢弃。

**修复**：`history.extend(result_messages)` 全量写入后再追加 final_reply（去重）。

### 🐛 Agent 节点 sync/async 不匹配

**问题**：`graph.invoke()` 报错 `No synchronous function provided to "product_agent"`。

**根因**：LangGraph 节点是 `async def`，但调用了同步的 `graph.invoke()` 而非 `graph.ainvoke()`。

**修复**：`execution/chat_service.py` — `graph.invoke()` → `await graph.ainvoke()`。

---

## RAG 稠密向量改造

### 目的
TF-IDF 是关键词匹配，不懂语义。"哈士奇不爱吃饭"和"食欲不振"得分一样。改为 embedding 稠密向量实现语义级检索。

### 实现
- 模型：`paraphrase-multilingual-MiniLM-L12-v2`（384维，420MB，本地加载）
- 改 `cognition/engine.py`：检索从 TF-IDF 余弦相似度 → embedding 点积语义匹配
- 模型加载失败自动降级 TF-IDF
- 缓存 `cache/embeddings.npy`，重启秒加载

### 效果
"哈士奇不爱吃饭怎么办" → 匹配到 `dog_feeding.md` 中"食欲下降、换粮方法"相关内容，语义理解质变。

---

### V4 — 摘要压缩 LLM 调用优化（已实施）

**问题**：每次对话超 6000 token 就调 LLM 压缩历史，单次 11 秒，占请求总耗时 45%。

**根因**：`token_based_trim` → `_summarize_history` → DeepSeek LLM 调用无超时，偶发长尾延迟。

**修复**：
- `config/settings.py` — `MAX_HISTORY_TOKENS` 6000→12000，`RECENT_WINDOW_TOKENS` 2500→4000
- `execution/chat_service.py` — `_get_client()` 加 `timeout=30.0, max_retries=1`

**效果**：单次请求 25s → 8-10s（降 ~60%）
