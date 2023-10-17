from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Create a callback handler


class MyCallbackHandler(StreamingStdOutCallbackHandler):
    def on_llm_new_token(self, *args, **kwargs):
        print(f'Args: {args}')
        print(f'Kwargs: {kwargs}')
        token = kwargs.get('token')  # updated line to get 'token' from kwargs

        print('token is', token)


# Instantiate a ChatOpenAI object with streaming enabled and your custom callback handler
chat = ChatOpenAI(
    streaming=True,
    callbacks=[MyCallbackHandler()]
)

# Send a message and get a streamed response
response = chat([HumanMessage(content="Hello, how are you?")])
