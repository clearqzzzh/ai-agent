"""最简测试：验证消息在上、输入在下"""
import streamlit as st

if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "pending" not in st.session_state:
    st.session_state.pending = None

# ① 处理上轮提交
if st.session_state.pending:
    st.session_state.msgs.append(st.session_state.pending)
    st.session_state.pending = None

# ② 渲染消息
for m in st.session_state.msgs:
    with st.chat_message("user"):
        st.write(m)

# ③ 输入栏
p = st.chat_input("输入...")
if p:
    st.session_state.pending = p
    st.rerun()
