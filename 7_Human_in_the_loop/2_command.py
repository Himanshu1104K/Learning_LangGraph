from langgraph.graph import StateGraph, add_messages, END
from langgraph.types import Command
from typing import TypedDict


class State(TypedDict):
    text: str


def node_a(state: State):
    print("Node A")
    return Command(goto="node_b", update={"text": state["text"] + "a"})


def node_b(state: State):
    print("Node B")
    return Command(goto="node_c", update={"text": state["text"] + "b"})


def node_c(state: State):
    print("Node C")
    return Command(goto="node_d", update={"text": state["text"] + "c"})


def node_d(state: State):
    print("Node D")
    return Command(goto=END, update={"text": state["text"] + "d"})


print("Starting the graph")

graph = StateGraph(State)

graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)
graph.add_node("node_d", node_d)

graph.set_entry_point("node_a")

app = graph.compile()

response = app.invoke({"text": ""})

print(response)
