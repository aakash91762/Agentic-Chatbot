from Learnings.agentic_chatbot_backend import chatbot 
from langchain_core.messages import BaseMessage, HumanMessage


thread_id = "2"
initial_state = {
    'messages': [HumanMessage(content='what is python generate a blog?')]
}

config = {'configurable': {'thread_id': thread_id}}
# response = chatbot.invoke(initial_state, config=config)
# print(response['messages'][-1].content)
for chunk,metadata in chatbot.stream(
    initial_state,
    config=config,
    stream_mode=initial_state
):   
    if chunk.content:
        print(chunk.content,end="",flush=True)