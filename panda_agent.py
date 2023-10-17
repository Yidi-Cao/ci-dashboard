from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
# from langchain.callbacks import StreamHandler
from langchain.callbacks import StdOutCallbackHandler
import pandas as pd

df = pd.read_csv("./data/frog_taco.csv")

# filtered_set = df.loc[(df['观点'] == '牛蛙味道辣') & (
#     df['情感倾向'] == '负面')]['评论原文'].drop_duplicates()

# print(filtered_set)

agent = create_pandas_dataframe_agent(
    ChatOpenAI(api_key="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",
               temperature=0, model="gpt-4"),
    df,
    verbose=False,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

foo = agent.run("给我filter出关于牛蛙味道辣的负面评论")
print("foo", foo)
