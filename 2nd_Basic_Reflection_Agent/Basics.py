from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.graph import END, MessageGraph
from Chains import generation_chain, reflection_chain

load_dotenv()

graph = MessageGraph()

REFLECT = "reflect"
GENERATE = "generate"


def generate_node(state):
    return generation_chain.invoke({"messages": state})


def reflect_node(state):
    response = reflection_chain.invoke({"messages": state})
    return [HumanMessage(content=response.content)]


graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

# Set the entry point
graph.set_entry_point(GENERATE)


def should_continue(state):
    if len(state) > 4:
        return "end"
    return "reflect"


# Add the conditional edge with explicit path_map
graph.add_conditional_edges(GENERATE, should_continue, {"reflect": REFLECT, "end": END})

# Add the edge from REFLECT back to GENERATE
graph.add_edge(REFLECT, GENERATE)

# Compile the graph
app = graph.compile()

# Print both visualizations to debug
print("MERMAID DIAGRAM:")
print(app.get_graph().draw_mermaid())
print("\nASCII VISUALIZATION:")
app.get_graph().print_ascii()

response = app.invoke([HumanMessage(content="AI AGENTS TAKING OVER CONTENT CREATION")])
print(response)