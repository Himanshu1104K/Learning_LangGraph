from typing import TypedDict, Annotated
from langgraph.graph import END, add_messages, StateGraph
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

llm = ChatGroq(model_name="llama-3.1-8b-instant")

sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)

memory = SqliteSaver(sqlite_conn)
window_size = 3


class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: BasicChatState):
    # Limit the message history to window_size messages
    if (
        len(state["messages"]) > window_size * 2
    ):  # *2 because each exchange has user+AI messages
        state["messages"] = state["messages"][-window_size * 2 :]

    return {"messages": [llm.invoke(state["messages"])]}


graph = StateGraph(BasicChatState)
graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": 1}}

while True:
    user_input = input("User : ")
    if user_input.lower() == "exit":
        break
    else:
        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )
        print(result["messages"][-1].content)
