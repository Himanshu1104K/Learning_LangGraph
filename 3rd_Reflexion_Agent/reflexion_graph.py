from typing import List

from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools

graph = MessageGraph()
MAX_ITERATIONS = 2

graph.add_node("draft", first_responder_chain)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revisor_chain)


graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revisor")


def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return "end"
    return "execute_tools"


graph.add_conditional_edges(
    "revisor", event_loop, {"end": END, "execute_tools": "execute_tools"}
)
graph.set_entry_point("draft")

app = graph.compile()

print(app.get_graph().draw_mermaid())
print(app.get_graph().draw_ascii())

# Properly format the input as a list of messages
input_messages = [HumanMessage(content="Write about how small business can leverage AI to grow")]
response = app.invoke(input_messages)

print(response[-1].tool_calls[0]["args"]["answer"])
print(response, "response")
