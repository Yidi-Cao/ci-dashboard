import streamlit as st
import openai
import pandas as pd
import numpy as np

# 设置你的OpenAI的API密钥
openai.api_key = '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'

st.sidebar.markdown("# chatbot page 🤖 ")
st.title("Travis智能助理")
st.write("I am Travis, your personal assistant. How can I help you today?")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("what's up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # with st.chat_message("user"):
    #     st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state.openai_model,
            messages=[{"role": m["role"], "content": m["content"]}
                      for m in st.session_state.messages],
            stream=True
        ):
            full_response += response["choices"][0].delta.get(
                "content", "")
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})
