from typing import TypedDict, Annotated, List
from langgraph.graph import END, add_messages, StateGraph
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langchain_chroma import Chroma
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import uuid

load_dotenv()
MAX_WINDOW_SIZE = 10  # Increased for better context retention

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# Initialize the vector store with an embedding function
vector_store = Chroma(
    collection_name="conversation_history",
    embedding_function=embeddings,
    persist_directory="db",
)


class BasicChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


llm = ChatAnthropic(model="claude-3-7-sonnet-20250219")

search_tool = TavilySearchResults(max_results=4)
tools = [search_tool]

llm_with_tools = llm.bind_tools(tools=tools)


def get_similar_from_recent(query, n_results=5):
    """Retrieve relevant conversation history using semantic search."""
    try:
        all_documents = vector_store.get()

        if not all_documents or len(all_documents["documents"]) == 0:
            return []

        # Sort documents by timestamp (newest first)
        sorted_docs = sorted(
            zip(
                all_documents["documents"],
                all_documents["metadatas"],
                all_documents["ids"],
            ),
            key=lambda x: x[1]["timestamp"],
            reverse=True,
        )

        # Take only the recent messages based on MAX_WINDOW_SIZE
        recent_docs = sorted_docs[:MAX_WINDOW_SIZE]
        recent_ids = [doc[2] for doc in recent_docs]

        # First try with similarity search if we have enough documents
        if recent_ids:
            similar_docs = vector_store.similarity_search(
                query, k=n_results, filter={"id": {"$in": recent_ids}}
            )
            return similar_docs
        else:
            return []

    except Exception as e:
        return []


def store_message(content, role="human"):
    """Store a message in the vector store with proper metadata."""
    try:
        doc_id = str(uuid.uuid4())
        timestamp = time.time()

        vector_store.add_documents(
            [
                Document(
                    page_content=content,
                    metadata={"timestamp": timestamp, "role": role, "id": doc_id},
                )
            ],
            ids=[doc_id],
        )
    except Exception:
        pass


def chatbot(state: BasicChatState):
    """Process user query with conversation memory."""
    # Get the user's query
    query = state["messages"][-1].content

    # Store the user's message in the vector store
    store_message(query, role="human")

    # Retrieve relevant conversation history
    context_docs = get_similar_from_recent(query)

    # Format context for the prompt
    if context_docs:
        conversation_history = []
        for doc in context_docs:
            role = doc.metadata.get("role", "unknown")
            content = doc.page_content
            conversation_history.append(f"{role.upper()}: {content}")

        context_str = "\n".join(conversation_history)
    else:
        context_str = "No relevant conversation history found."

    # Create a more structured prompt with clear sections
    prompt = f"""Answer the user's question or respond to their message.

CONVERSATION HISTORY:
{context_str}

CURRENT USER QUESTION:
{query}

Based on the conversation history and the current question, provide a helpful response. 
If the question asks about information that was mentioned in previous messages, be sure to recall and use that information.
"""

    # Get response from the LLM
    response = llm_with_tools.invoke(prompt)

    # Extract content from the response
    if hasattr(response, "content"):
        if isinstance(response.content, str):
            content = response.content
        elif isinstance(response.content, list):
            content = " ".join(
                [
                    item.get("text", "")
                    for item in response.content
                    if isinstance(item, dict)
                ]
            )
        else:
            content = str(response.content)
    else:
        content = str(response)

    # Store the AI's response
    store_message(content, role="ai")

    return {"messages": [response]}


def tools_router(state: BasicChatState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END


tool_node = ToolNode(tools=tools)

workflow = StateGraph(BasicChatState)
workflow.add_node("chatbot", chatbot)
workflow.add_node("tool_node", tool_node)
workflow.set_entry_point("chatbot")
workflow.add_conditional_edges("chatbot", tools_router)
workflow.add_edge("tool_node", "chatbot")

graph = workflow.compile()

while True:
    user_input = input("User : ")
    if user_input.lower() == "exit":
        break
    else:
        result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
        print(result["messages"][-1].content)
