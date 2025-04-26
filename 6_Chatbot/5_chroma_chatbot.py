from typing import List, TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import uuid
import time
import chromadb
import warnings
from langgraph.graph import END, StateGraph, add_messages

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, "db", "chroma_db")
os.makedirs(persistent_dir, exist_ok=True)

llm = ChatGroq(model_name="llama-3.1-8b-instant")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB with direct client
client = chromadb.PersistentClient(path=persistent_dir)
# Create or get collection
try:
    collection = client.get_or_create_collection("chat_history")
except:
    # If collection exists but is corrupted, recreate it
    try:
        client.delete_collection("chat_history")
    except:
        pass
    collection = client.create_collection("chat_history")

# Initialize LangChain's Chroma wrapper
db = Chroma(
    client=client, collection_name="chat_history", embedding_function=embeddings
)

# Settings
MAX_CHAT_HISTORY = 2  # Maximum number of chat exchanges to store


# Define the state of the chatbot
class ChatState(TypedDict):
    messages: Annotated[List, add_messages]
    user_message: str
    similar_context: str


def store_message(content, role):
    """Store a message in ChromaDB"""
    # Generate a unique ID
    doc_id = str(uuid.uuid4())

    # Add to ChromaDB with timestamp
    db.add_texts(
        texts=[content],
        metadatas=[{"role": role, "timestamp": time.time()}],
        ids=[doc_id],
    )

    # Ensure we keep only the last MAX_CHAT_HISTORY exchanges
    limit_chroma_history()


def limit_chroma_history():
    """Ensure only the last MAX_CHAT_HISTORY exchanges are kept in ChromaDB"""
    try:
        results = db.get()
    except:
        return

    if not results or not results["ids"] or len(results["ids"]) <= MAX_CHAT_HISTORY * 2:
        return

    # Sort documents by timestamp
    documents = []
    for i, doc_id in enumerate(results["ids"]):
        content = results["documents"][i]
        metadata = results["metadatas"][i]
        timestamp = metadata.get("timestamp", 0)
        documents.append((doc_id, timestamp))

    # Sort by timestamp
    documents.sort(key=lambda x: x[1])

    # Calculate how many to remove (keeping only the most recent ones)
    to_remove = len(documents) - (MAX_CHAT_HISTORY * 2)
    if to_remove <= 0:
        return

    # Get IDs of oldest documents to remove
    ids_to_remove = [doc[0] for doc in documents[:to_remove]]

    # Remove them from ChromaDB
    db.delete(ids=ids_to_remove)


def get_context_from_similar_messages(query):
    """Get context from similar past messages"""
    # Use the retriever pattern
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    try:
        # Get similar messages using the new invoke method
        similar_results = retriever.invoke(query)

        # Create context from similar past conversations
        context = ""
        if similar_results:
            context = "Here are some relevant previous conversations:\n"
            for i, doc in enumerate(similar_results):
                context += f"- {doc.page_content}\n"

        return context
    except Exception as e:
        print(f"Warning: Could not retrieve similar messages: {str(e)}")
        return ""


# LangGraph node functions
def process_user_message(state: ChatState):
    """Process the user message, store it, and find similar context"""
    # Extract user message from messages
    user_message = state["messages"][-1].content

    # Store user message in ChromaDB
    store_message(user_message, "user")

    # Find similar messages for context
    similar_context = get_context_from_similar_messages(user_message)

    # Return updated state
    return {"user_message": user_message, "similar_context": similar_context}


def generate_response(state):
    """Generate a response based on the conversation history and context"""
    # Create augmented messages with context
    augmented_messages = state["messages"].copy()

    # If we have context, add it as a system message at the beginning
    if state["similar_context"]:
        # Find position to insert system message (after any existing system messages)
        system_msg_index = 0
        for i, msg in enumerate(augmented_messages):
            if isinstance(msg, SystemMessage):
                system_msg_index = i + 1
            else:
                break

        # Insert context as a system message
        augmented_messages.insert(
            system_msg_index, SystemMessage(content=state["similar_context"])
        )

    # Generate AI response
    ai_response = llm.invoke(augmented_messages)

    # Store AI response in ChromaDB
    store_message(ai_response.content, "assistant")

    # Return updated state with AI response added to messages
    return {"messages": [ai_response]}


# Create the graph
workflow = StateGraph(ChatState)

# Add nodes
workflow.add_node("process_message", process_user_message)
workflow.add_node("generate_response", generate_response)

# Add edges
workflow.set_entry_point("process_message")
workflow.add_edge("process_message", "generate_response")
workflow.add_edge("generate_response", END)

# Compile the graph
app = workflow.compile()

print("Chat started. Type 'exit' to end, 'clear' to start fresh.")
print("The system will use context from past conversations to improve responses.")
print("---------------------------------------------------------------------")

while True:
    user_input = input("User : ")

    if user_input.lower() == "exit":
        break

    elif user_input.lower() == "clear":
        # Clear the ChromaDB collection
        try:
            all_ids = db.get()["ids"]
            if all_ids:
                db.delete(ids=all_ids)
            print("Conversation history cleared.")
        except Exception as e:
            print(f"Error clearing history: {str(e)}")

    else:
        try:
            # Create initial state with user message
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "user_message": "",
                "similar_context": "",
            }

            # Invoke the graph
            result = app.invoke(initial_state)

            # Display result
            print(result["messages"][-1].content)

        except Exception as e:
            print(f"Error: {str(e)}")
            print("Let's try again.")

print("Chat ended.")
