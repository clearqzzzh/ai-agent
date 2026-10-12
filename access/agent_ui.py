"""
Streamlit 聊天界面 —— 宠物商城智能导购（Qwen 风格）
启动：cd D:/kgezhuzhi/agent-fenqu/agent-service && ../.venv/Scripts/streamlit run access/agent_ui.py
"""
import json, time, base64, hashlib
import requests, streamlit as st, urllib.parse

# ═══════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════
API_URL = "http://localhost:8001/chat"
TYPING_SPEED = 0.012

# 快捷模板（只保留订单）
TEMPLATES = [
    ("📦 我的订单", "帮我查看一下最近的订单"),
]

# ═══════════════════════════════════════════════════════
#  页面设置
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="宠物商城智能导购",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════
#  自定义 CSS（纯白背景，消除白雾）
# ═══════════════════════════════════════════════════════
st.markdown("""<style>
    /* ── 全局纯白（不影响头像）── */
    html, body, .stApp, .main, .block-container,
    section[data-testid="stMain"], section[data-testid="stMain"] > div,
    [data-testid="stVerticalBlock"], [data-testid="stVerticalBlock"] > div {
        background: #ffffff !important;
        background-image: none !important;
    }
    /* 清除伪元素 */
    .stApp::before, .stApp::after,
    .main::before, .main::after,
    .block-container::before, .block-container::after,
    [data-testid="stVerticalBlock"]::before, [data-testid="stVerticalBlock"]::after,
    [data-testid="stVerticalBlock"] > div::before, [data-testid="stVerticalBlock"] > div::after {
        display: none !important;
        content: none !important;
    }

    @media (max-width: 640px) {
        [data-testid="stSidebar"] { display: block !important; }
        [data-testid="collapsedControl"] { display: none !important; }
    }

    /* ── 气泡样式（基于 CSS 类名，不依赖选择器）── */
    [data-testid="stChatMessageContent"] {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
    }
    .pet-user-bubble {
        background: linear-gradient(135deg, #4A90D9, #357ABD) !important;
        color: #fff !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 10px 16px !important;
        max-width: 70% !important;
        width: fit-content !important;
        display: inline-block;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        font-size: 15px; line-height: 1.7;
    }
    .pet-user-bubble,
    .pet-user-bubble p,
    .pet-user-bubble span,
    .pet-user-bubble li,
    .pet-user-bubble h1,
    .pet-user-bubble h2,
    .pet-user-bubble h3,
    .pet-user-bubble h4 {
        color: #fff !important;
    }
    .pet-ai-bubble {
        background: #fff !important;
        color: #333 !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 10px 16px !important;
        max-width: 80% !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
        border: 1px solid #e8e8e8 !important;
        display: inline-block;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        font-size: 15px; line-height: 1.7;
    }
    /* 聊天气泡内图片紧凑显示 */
    .pet-user-bubble img, .pet-ai-bubble img {
        max-width: 120px !important;
        width: 120px !important;
        height: auto !important;
        border-radius: 8px;
        display: block;
    }
    #MainMenu, footer, header[data-testid="stHeader"] { display: none; }
    [data-testid="stStatusWidget"],
    .stApp [data-testid="stVerticalBlock"] > div[style*="position"],
    iframe[title="streamlit_ace.widget"] ~ div { display: none !important; }
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  session_state 初始化
# ═══════════════════════════════════════════════════════
defaults = {
    "sid": None, "messages": [], "tool_log": [], "user_id": None,
    "pending_image": None, "_pending": None, "_sent_hash": "",
    "slist": [], "_skip_last": False, "username": "",
    "quick_q": None, "_typing_reply": None, "_typing_tools": None,
    "_user_submitted": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════
#  用户登录（URL 参数传入）
# ═══════════════════════════════════════════════════════
qp = st.query_params
if "info" in qp:
    try:
        info = json.loads(urllib.parse.unquote(qp["info"]))
        resp = requests.get(
            "http://localhost:8083/user/lookup",
            params={"phone": info.get("phone", "")}, timeout=5,
        )
        if resp.ok and resp.json().get("success"):
            st.session_state.user_id = resp.json()["data"]["id"]
            st.session_state.username = info.get("username", "")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════
#  标题
# ═══════════════════════════════════════════════════════
st.markdown(
    '<h1 style="text-align:center;">🐱 宠物商城智能导购</h1>',
    unsafe_allow_html=True,
)
st.caption("由 DeepSeek + RAG 知识库驱动  ·  支持图片识别与喂养知识问答")


# ═══════════════════════════════════════════════════════
#  历史会话（可折叠）
# ═══════════════════════════════════════════════════════
with st.expander("📚 历史会话", expanded=False):
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        if st.button("🔄 刷新列表", key="rh"):
            try:
                st.session_state.slist = requests.get(
                    "http://localhost:8001/sessions", timeout=3,
                ).json().get("sessions", [])
            except Exception:
                pass
    with c2:
        if st.button("🆕 新建会话", key="nc"):
            st.session_state.update(sid=None, messages=[], tool_log=[])
            st.rerun()
    with c3:
        if st.session_state.user_id:
            st.success(f"👤 ID:{st.session_state.user_id}")
        else:
            st.caption("未登录")

    if st.session_state.slist:
        idx = st.selectbox(
            "选择", range(len(st.session_state.slist)),
            format_func=lambda i: (
                f"{st.session_state.slist[i].get('preview', '')[:20]} "
                f"({st.session_state.slist[i]['session_id'][:6]})"
            ),
        )
        cl, cd = st.columns([3, 1])
        with cl:
            if st.button("📂 加载", key="ld"):
                s = st.session_state.slist[idx]
                r = requests.get(
                    f"http://localhost:8001/sessions/{s['session_id']}", timeout=3,
                )
                if r.ok:
                    st.session_state.update(
                        sid=s["session_id"],
                        messages=r.json().get("messages", []),
                        tool_log=[],
                    )
                    st.rerun()
        with cd:
            if st.button("🗑️", key="dl"):
                s = st.session_state.slist[idx]
                requests.delete(
                    f"http://localhost:8001/sessions/{s['session_id']}", timeout=3,
                )
                st.session_state.slist = [
                    x for x in st.session_state.slist
                    if x["session_id"] != s["session_id"]
                ]
                st.rerun()
    else:
        st.caption("点击「刷新列表」加载历史会话")

st.divider()


# ═══════════════════════════════════════════════════════
#  侧边栏
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.subheader("💬 当前对话")
    if st.session_state.messages:
        for m in st.session_state.messages[-6:]:
            icon = "👤" if m["role"] == "user" else "🐱"
            st.caption(f"{icon} {str(m.get('content', ''))[:25]}")
    else:
        st.caption("暂无对话")

    st.divider()
    st.subheader("🔧 工具调用")
    if st.session_state.tool_log:
        for t in st.session_state.tool_log[-8:]:
            st.caption(f"▸ {t}")
    else:
        st.caption("暂无")


# ═══════════════════════════════════════════════════════
#  核心逻辑
# ═══════════════════════════════════════════════════════

def call_agent(text: str) -> dict:
    """调用后端 /chat 接口"""
    body = {
        "session_id": st.session_state.sid,
        "user_id": st.session_state.user_id,
        "messages": [{"role": "user", "content": text}],
    }
    try:
        r = requests.post(API_URL, json=body, timeout=90)
        return r.json() if r.ok else {"reply": "服务异常"}
    except Exception:
        return {"reply": "连接失败"}


def _render_user_msg(msg: dict):
    """渲染用户消息（蓝色气泡，内联 HTML）"""
    c = msg.get("content", "")
    html = ""
    if msg.get("image"):
        html += f'<img src="data:image/jpeg;base64,{msg["image"]}" style="max-width:120px;width:120px;border-radius:8px;margin-bottom:6px;display:block;">'
    text = ""
    if isinstance(c, str) and c:
        text = c
    elif isinstance(c, list):
        text = " ".join(p.get("text", "") for p in c if p.get("type") == "text")
    if text:
        html += f"<p style='margin:0'>{text}</p>"
    st.markdown(f'<div class="pet-user-bubble">{html}</div>', unsafe_allow_html=True)


# ── ① 渲染历史消息 ──
msgs = list(st.session_state.messages)
typing_reply = st.session_state._typing_reply
typing_tools = st.session_state._typing_tools

# 有打字动画时（来自上一次交互），排除最后一条
if typing_reply is not None and msgs:
    msgs = msgs[:-1]

for msg in msgs:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            _render_user_msg(msg)
        else:
            c = msg.get("content", "")
            if isinstance(c, str) and c:
                st.markdown(f'<div class="pet-ai-bubble">{c}</div>', unsafe_allow_html=True)
            if isinstance(c, list):
                for p in c:
                    if p.get("type") == "text" and p.get("text"):
                        st.markdown(f'<div class="pet-ai-bubble">{p["text"]}</div>', unsafe_allow_html=True)
                    elif p.get("type") == "image_url":
                        st.image(p["image_url"]["url"], width=120)


# ── ② 处理 pending 请求 + 立即渲染打字动画 ──
if st.session_state._pending and st.session_state._user_submitted:
    pp = st.session_state._pending
    st.session_state._pending = None
    st.session_state._user_submitted = False
    prompt, img = pp["text"], pp.get("image")

    if img:
        # 有图片 → 调用千问 VL 识别
        st.session_state._sent_hash = hashlib.md5(img["bytes"]).hexdigest()
        img_url = f"data:{img['mime']};base64,{img['b64']}"
        try:
            r = requests.post(
                API_URL.replace("/chat", "/vision"),
                json={"image_url": img_url, "question": prompt or "请描述这张图片中的宠物品种"},
                timeout=15,
            )
            if r.ok:
                analysis = r.json().get("analysis", "")
                reply = analysis if analysis else "图片识别未返回结果，请换一张图片试试"
            else:
                reply = f"图片识别服务暂时不可用（状态码: {r.status_code}），请稍后重试"
        except requests.Timeout:
            reply = "图片识别超时，请换一张较小的图片重试"
        except Exception:
            reply = "图片识别出错，请稍后重试或直接描述您的问题"
        tools = []
    else:
        # 纯文本 → 调用 Agent
        data = call_agent(prompt)
        if data.get("session_id"):
            st.session_state.sid = data["session_id"]
        reply = data.get("reply", "")
        tools = data.get("tool_calls_made", [])

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.tool_log.extend(tools)

    # 立即渲染打字动画（不依赖下一轮 rerun）
    with st.chat_message("assistant"):
        ph = st.empty()
        d = ""
        for ch in reply:
            d += ch
            ph.markdown(f'<div class="pet-ai-bubble">{d}▍</div>', unsafe_allow_html=True)
            time.sleep(TYPING_SPEED)
        ph.markdown(f'<div class="pet-ai-bubble">{d}</div>', unsafe_allow_html=True)
        if tools:
            st.caption("🔧 " + ", ".join(tools))


# ── ③ 处理上一轮遗留的打字动画（仅当 ② 未执行时）──
elif typing_reply is not None:
    st.session_state._typing_reply = None
    st.session_state._typing_tools = None
    with st.chat_message("assistant"):
        ph = st.empty()
        d = ""
        for ch in typing_reply:
            d += ch
            ph.markdown(f'<div class="pet-ai-bubble">{d}▍</div>', unsafe_allow_html=True)
            time.sleep(TYPING_SPEED)
        ph.markdown(f'<div class="pet-ai-bubble">{d}</div>', unsafe_allow_html=True)
        if typing_tools:
            st.caption("🔧 " + ", ".join(typing_tools))


# ── 快捷问题自动填充 ──
quick_text = st.session_state.quick_q
if quick_text:
    st.session_state.quick_q = None


# ── ④ 底部输入栏 ──
# 布局：
#   ┌──────────────────────────────────────────────────┐
#   │ [图片预览✕]（有图才显示）                          │
#   ├──────────────────────────────────────────────────┤
#   │ [📎]  输入问题，按 Enter 发送...   [模板1][模板2]… │
#   └──────────────────────────────────────────────────┘

# 图片预览条（有图才显示）
if st.session_state.pending_image:
    pi_c1, pi_c2, _ = st.columns([0.5, 0.5, 9])
    with pi_c1:
        st.image(st.session_state.pending_image["bytes"], width=48, caption="待发送")
    with pi_c2:
        if st.button("✕", key="cx"):
            st.session_state.pending_image = None
            st.rerun()

# 输入行：📎上传 | 输入框 | 模板按钮
n_tpl = len(TEMPLATES)
col_widths = [0.4, 5.6] + [1] * n_tpl
input_row = st.columns(col_widths, gap="small")

# 📎 上传按钮
with input_row[0]:
    with st.popover("📎", use_container_width=True):
        st.caption("上传图片 (jpg/png/webp)")
        uf = st.file_uploader(
            "选择图片",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
            key="uf",
        )
        if uf:
            ib = uf.read()
            h = hashlib.md5(ib).hexdigest()
            if h != st.session_state._sent_hash:
                st.session_state.pending_image = {
                    "bytes": ib,
                    "mime": uf.type or "image/jpeg",
                    "b64": base64.b64encode(ib).decode(),
                }
                st.success("✅ 已选择图片")
                st.image(ib, width=180)
            else:
                st.caption("⚠️ 该图片已发送过")

# 聊天输入框
with input_row[1]:
    prompt = st.chat_input("输入问题，按 Enter 发送...")

# 模板按钮（输入框右边）
for i, (label, query) in enumerate(TEMPLATES):
    with input_row[2 + i]:
        if st.button(label, key=f"tpl_{i}", use_container_width=True):
            st.session_state.quick_q = query

# ── ⑤ 发送处理（两步状态机）──
if prompt or quick_text:
    text = prompt or quick_text
    img = st.session_state.pending_image
    st.session_state.pending_image = None

    # 构建用户消息（含图片 base64 用于回显）
    user_msg = {"role": "user", "content": text}
    if img:
        user_msg["image"] = img["b64"]
    st.session_state.messages.append(user_msg)

    # 第一步：标记用户已提交，rerun 后用户消息先在 ① 中渲染
    st.session_state._pending = {"text": text, "image": img}
    st.session_state._user_submitted = True
    st.rerun()
