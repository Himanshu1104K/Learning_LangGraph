from langgraph.graph import StateGraph, END, START, add_messages
from langgraph.types import interrupt, Command
from typing import Annotated, TypedDict, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import uuid

llm = ChatGroq(model_name="llama3-8b-8192")


class State(TypedDict):
    linkedin_topic: str
    generated_post: Annotated[List[str], add_messages]
    human_feedback: Annotated[List[str], add_messages]


def model(state: State):
    """Here we are using llm to generate a linkedin post with human feedback incorporated"""

    print("[Model] Generating Content")
    linkedin_topic = state["linkedin_topic"]
    feedback = (
        state["human_feedback"] if "human_feedback" in state else ["No feedback yet"]
    )

    # Here we define the prompt

    prompt = f"""
    Linkedin Topic : {linkedin_topic}
    Human Feedback : {feedback[-1] if feedback else "No feedback yet"}

    Generate a structured & well written linkedin post based on the given topic

    Consider previous human feedback to refine the result.
    """

    # Here we generate the post
    response = llm.invoke(
        [
            SystemMessage(content="You are an expert Linkedin Content Writer"),
            HumanMessage(content=prompt),
        ]
    )

    generated_linkedin_post = response.content

    print(f"[Model] Generated Post : {generated_linkedin_post}")

    return {"generated_post": generated_linkedin_post, "human_feedback": feedback}


def human_node(state: State):
    """Human intervention node - loops back to model unless input is done."""

    print("\n [Human Node] awaiting human feedback.")

    generated_post = state["generated_post"]

    user_feedback = interrupt(
        {
            "generated_post": generated_post,
            "message": "Provide feedback or type 'done' to finish",
        }
    )

    print(f"\n [Human Node] Received Feedback : {user_feedback}")

    if user_feedback.lower() in ["done", "finish"]:
        return Command(
            goto="end_node",
            update={"human_feedback": state["human_feedback"] + ["Finalized"]},
        )
    else:
        return Command(
            goto="model",
            update={"human_feedback": state["human_feedback"] + [user_feedback]},
        )


def end_node(state: State):
    """Final Node"""
    print("\n [End Node] Process finished...")
    print("Final Generated Post :", state["generated_post"][-1])
    print("Final Human Feedback :", state["human_feedback"])

    return {
        "generated_post": state["generated_post"],
        "human_feedback": state["human_feedback"],
    }


# Building the graph
graph = StateGraph(State)

graph.add_node("model", model)
graph.add_node("human_node", human_node)
graph.add_node("end_node", end_node)

graph.set_entry_point("model")

graph.add_edge(START, "model")
graph.add_edge("model", "human_node")

graph.set_finish_point("end_node")

checkpointer = MemorySaver()

app = graph.compile(checkpointer=checkpointer)

thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

linkedin_topic = input("Enter a LinkedIn Topic : ")

initial_state = {
    "linkedin_topic": linkedin_topic,
    "generated_post": [],
    "human_feedback": [],
}

from IPython.display import Image, display

display(Image(app.get_graph().draw_mermaid_png()))

for chunk in app.stream(initial_state, config=thread_config):
    for node_id, value in chunk.items():
        if node_id == "__interrupt__":
            while True:
                user_feedback = input("Enter your feedback or type 'done' to finish: ")

                app.invoke(Command(resume=user_feedback), config=thread_config)

                if user_feedback.lower() in ["done", "finish"]:
                    break
