from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
import operator


class SimpleState(TypedDict):
    count: int
    sum: Annotated[int, operator.add]
    history: Annotated[List[int], operator.concat]


def increment(state: SimpleState) -> SimpleState:
    count = state["count"] + 1
    return {
        "count": count,
        "sum": count,
        "history": [count],
    }


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

state = {"count": 0, "sum": 0, "history": []}
result = app.invoke(state)

print(result)
