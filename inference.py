import time
import openai
import streamlit as st

def gpt_query(openai, prompt, engine, temperature=0, max_tokens=8000, top_p=0):
    now = time.time()
    messages = [
        {"role": "system",
         "content": prompt["system"]
         },
        {
            "role": "user",
            "content": prompt['user']
        }
    ]
    response = openai.ChatCompletion.create(
        engine=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=0,
        presence_penalty=0)

    time_used = time.time() - now
    return response, time_used


def gpt_query_stream(client, messages, temperature=0, max_tokens=8000, top_p=0):
    for chunk in client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=True
    ):
        if chunk.choices and len(chunk.choices) > 0:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content


def init_gpt_4():
    engine = "gpt-4"
    # openai.api_base = "https://yumc-et-azure-openai.openai.azure.com/"
    # openai.api_type = "azure"
    # openai.api_version = "2023-07-01-preview"
    # openai.api_key = "90c9e8bcffc040fc9e2ddf0479309f7b"
    api_key = st.secrets["OPENAI_API_KEY"]
    openai.api_key = api_key


    return openai, engine
