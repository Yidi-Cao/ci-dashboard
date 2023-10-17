import time
import openai


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


def gpt_query_stream(openai, prompt, engine, temperature=0, max_tokens=8000, top_p=0):
    messages = [
        {"role": "system",
         "content": prompt["system"]
         },
        {
            "role": "user",
            "content": prompt['user']
        }
    ]
    now = time.time()
    for chunk in openai.ChatCompletion.create(
        engine=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=0,
        presence_penalty=0,
        stream=True
    ):
        print('chunk', chunk)
        if (chunk['choices'] and len(chunk['choices']) > 0):
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content is not None:
                time_used = time.time() - now
                print("time_used = time.time() - now", time_used)
                yield content


def init_gpt_4():
    engine = "gpt-4-32k"
    openai.api_base = "https://yumc-et-azure-openai.openai.azure.com/"
    openai.api_type = "azure"
    openai.api_version = "2023-07-01-preview"
    openai.api_key = "90c9e8bcffc040fc9e2ddf0479309f7b"
    # openai.api_key = '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'

    return openai, engine
