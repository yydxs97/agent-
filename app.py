import time

import streamlit as st
from agent.react_agent import ReactAgent
from model.factory import chat_model, embed_model  # ← 新增导入

# 🔍 启动时打印模型信息（仅首次加载显示）
if "model_verified" not in st.session_state:
    st.sidebar.info(f"""
    **当前接入的大模型状态**:
    - Chat: {type(chat_model).__name__} | base_url={getattr(chat_model, 'openai_api_base', 'N/A')}
    - Embed: {type(embed_model).__name__} | model={getattr(embed_model, 'model', 'N/A')}
    """)
    st.session_state["model_verified"] = True


# 标题
st.title("智扫通机器人智能客服")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt)

        def capture(generator, cache_list):

            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        st.rerun()
