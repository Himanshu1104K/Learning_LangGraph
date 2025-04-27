from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, add_messages, END
from langchain_groq import ChatGroq


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatGroq(model_name="llama3-8b-8192")

GENERATE_POST = "generate_post"
GET_REVIEW_DECISION = "get_review_decision"
POST = "post"
COLLECT_FEEDBACK = "collect_feedback"


def generate_post(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}


def get_review_decision(state: State):
    post_content = state["messages"][-1].content

    print("\n\nCurrent Linkedin Post \n\n")
    print(post_content)
    print("\n\n")

    decision = input("Post to Linkedin? (yes/no)")

    if decision == "yes":
        return POST
    else:
        return COLLECT_FEEDBACK


def post(state: State):
    final_post = state["messages"][-1].content

    print("\n\nFinal Linkedin Post \n\n")
    print(final_post)
    print("\nPost has been approved and posted to Linkedin")


def collect_feedback(state: State):
    feedback = input("How can I imporve the post?")

    return {"messages": [HumanMessage(content=feedback)]}


graph = StateGraph(State)

graph.add_node(GENERATE_POST, generate_post)
graph.add_node(GET_REVIEW_DECISION, get_review_decision)
graph.add_node(POST, post)
graph.add_node(COLLECT_FEEDBACK, collect_feedback)

graph.set_entry_point(GENERATE_POST)

graph.add_conditional_edges(GENERATE_POST, get_review_decision)
graph.add_edge(POST, END)
graph.add_edge(COLLECT_FEEDBACK, GENERATE_POST)

app = graph.compile()

response = app.invoke(
    {
        "messages": [
            HumanMessage(content="Write a post about the benefits of using LangGraph")
        ]
    }
)

print(response)
