from typing import TypedDict
from langgraph.graph import StateGraph, END


class SimpleState(TypedDict):
    count: int


def increment(state: SimpleState) -> SimpleState:
    return {"count": state["count"] + 1}


def should_continue(state: SimpleState):
    if state["count"] < 5:
        return "continue"
    else:
        return "stop"


graph = StateGraph(SimpleState)

graph.add_node("increment", increment)
graph.add_conditional_edges(
    "increment",
    should_continue,
    {"continue": "increment", "stop": END},
)

graph.set_entry_point("increment")

app = graph.compile()

result = app.invoke({"count": 0})

print(result)
