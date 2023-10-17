import streamlit as st
import openai
import pandas as pd
import numpy as np

# 设置你的OpenAI的API密钥
openai.api_key = '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'

option = st.selectbox(
    '选择你要查看的期货价格?',
    ['all', '棉花', '玉米', '大豆'])

'You selected: ', option

col1, col2 = st.columns(2)
with col1:
    chart_data = pd.DataFrame(
        50*np.abs(np.random.randn(20, 3)),
        columns=['棉花', '玉米', '大豆'])
    if option == 'all':
        st.line_chart(chart_data)
    else:
        st.line_chart(chart_data[option])

with col2:
    chart_data = pd.DataFrame(
        50 * np.abs(np.random.randn(20, 3)),
        columns=['棉花', '玉米', '大豆'])
    if option == 'all':
        st.bar_chart(chart_data)
    else:
        st.bar_chart(chart_data[option])


if option == 'all':
    st.table(chart_data)
else:
    st.table(chart_data[option])


df = pd.read_csv("./data/frog_taco.csv")

filtered_set = df.loc[(df['观点'] == '牛蛙味道辣') & (
    df['情感倾向'] == '负面')]['评论原文'].drop_duplicates()


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
    prompt_context = '''
    根据用户评论和问题，总结观点
    问题: {prompt}
    用户评论:
        {filtered_set}
    '''

    filled_prompt = prompt_context.format(
        prompt=prompt,
        filtered_set=filtered_set
    )

    st.session_state.messages.append(
        {"role": "user", "content": filled_prompt})

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
